import os
import importlib
import yaml
import pytest
import pandas as pd
from secTools import importDataTools, SecLoader, yamlLoad, upload_to_table
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import sqlalchemy
from sqlalchemy import create_engine, MetaData


class SECdataset(object):
	def __init__(self,name,cfg,num=None,pre=None,sub=None,tag=None):
		self.name=name
		self.cfg=cfg
		self.num=num
		self.pre=pre
		self.tag=tag
		self.sub=sub
		
		if num is None:
		#get the data from source.....
		
			#load sub table
			sub=SecTable(name='sub',cfg=cfg[0])
			
			#format sub table
			sub.df=self.format_sub(sub)
		
			#filter out the financial companies'
			sub.df=sub.df[(sub.df.sic>=7000)|(sub.df.sic<6000)]
			self.sub=sub
			
			#get the felevant filings codes
			relevant_filings=sub.df.adsh

			#create filter dictionary
			pre_filters={'adsh':relevant_filings}
			
			#load pre data
			pre=SecTable(name='pre',cfg=cfg[1],filters=pre_filters)	
			self.pre=pre
			
			relevant_tags=pre.df.tag.values.categories.tolist()
			
			#create filter to import tag information. 
			#Abstract =0 gets rid of non numerical figures
			#data type is monetary gets rid of anything that isnt monetary - like shares
			
			tag_filters={'tag':relevant_tags,'abstract':[0],'datatype':['monetary']}
			tag=SecTable(name='tag',cfg=cfg[2],filters=tag_filters)
			
			tag=self.format_tag(tag)
			self.tag=tag
					
			#update the relevant tags now 
			relevant_tags=self.tag.df.tag.values
			relevant_tags=relevant_tags.remove_unused_categories()
			
			#update pre table with relevant tags
			self.pre.df=self.pre.df[self.pre.df.tag.isin(relevant_tags)]
				
			#import num files according to relevant adsh and tag
			num_filter={'tag':relevant_tags,'adsh':relevant_filings}
			num=SecTable(name='num',cfg=cfg[3],filters=num_filter)
			num=self.format_num(num)

		
		self.num=num

	def format_sub(self,sub):
		df=sub.df
		df['cik']=df['cik'].astype(int)
		#some companies have no SIC code - they are mostly financials
		df.dropna(subset=['sic'],inplace=True)
		df['sic']=df['sic'].astype(int)
		df['period']=pd.to_datetime(df.period,format='%Y%m%d')
		df['filed']=pd.to_datetime(df.filed,format='%Y%m%d')
		
		return df

	def format_tag(self,tag):
		tag.df=tag.df.drop(axis=1,columns=['abstract','datatype'])
		tag.df.drop_duplicates(subset='tag',inplace=True)		
	
		return tag
	
	def format_num(self,num):
		#format dates
		num.df['period']=pd.to_datetime(num.df.ddate,format='%Y%m%d')
		
		#only keep USD values
		num.df=num.df[num.df['uom']=='USD']
		tf=num.df.duplicated(['adsh','tag','ddate','qtrs',],keep=False)
		print('There are',sum(tf), ' duplicated hetero values in the data which we will jettison')

		num.df=num.df[~tf]
		num.df=num.df.drop(axis=1,columns=['ddate','uom'])
		
		#define previous period
		num.df['prevper']= num.df['period'] - pd.DateOffset(years=1)

		#add previous period value as column in num table
		num.df=num.df.merge(num.df[['adsh','tag','period','value','qtrs']],left_on=['adsh','tag','qtrs','prevper'],
							right_on=['adsh','tag','qtrs','period'],how='left',suffixes=['','_p'])
		num.df.dropna(subset=['period_p'],inplace=True)
		num.df.drop(axis=1,columns=['prevper'],inplace=True)
		return num
	
		
	def ready_for_db_upload(self,inplace=False):
		num=self.num.df.copy()
		tag=self.tag.df.copy()
		sub=self.sub.df.copy()
		pre=self.pre.df.copy()

		#create an index for later data compression  lowest codes assigned to most common tags
		df=pd.DataFrame(num.groupby(['tag'])['value'].count().sort_values(ascending=False)).reset_index()
		tag=df.merge(tag,on='tag',how='left')
				
		dic={k:v for k,v in zip(tag.tag.values,tag.index.values)}

		#replace text tags with tag code
		num['tag_index']=num.tag.map(dic)
		pre['tag_index']=pre.tag.map(dic)

		num=num.drop(axis=1,columns='tag')
		pre=pre.drop(axis=1,columns='tag')


		if inplace:
			self.num.df=num
			self.pre.df=pre
			self.tag.df=tag
			self.sub.df=sub
		else:
			
			return num,tag,pre,sub
	
	
	def upload_to_db(self,db_location):
		engine=create_engine('sqlite:///'+db_location, echo=True)
		
		num,tag,pre,sub=self.ready_for_db_upload()
		
		upload_to_table(sub,'sub',engine)
		upload_to_table(pre,'pre',engine,batchnum=2000)
		upload_to_table(tag[['tag','iord','crdr','tlabel','value']],'tag',engine,batchnum=5000)
		upload_to_table(num,'num',engine,batchnum=2000)
	
	def tag_count(self):
	#deprecated
		relevant_tags=self.tag.tag
		df=self.pre[self.pre.tag.isin(relevant_tags)]
		tag_count=df[['tag','adsh']].groupby('tag').count().sort_values(by='adsh',ascending=False)
		return tag_count


	def tag_count_plot(self,inplace=False):
		tag_c=self.tag[['tag','value']].set_index('tag')
		tag_num=np.logspace(np.log2(tag_c.max()),2,num=20,base=2,dtype='int')
		tag_surv=[(tag_c[tag_c>p].count().value,p) for p in tag_num]
		sur_df=pd.DataFrame(tag_surv,columns=['distinct tag (features) num','number of appearances'])
		sur_df.plot(x='distinct tag (features) num',y='number of appearances',logy=True,logx=True,
					title='Feature survival rate',xlim=[1,2000])
		if inplace:
			self.sur_df=sur_df
			self.tag_c=tag_c
		else:
			return sur_df,tag_c
	
	def feature_prune(self,thresh=3000,inplace=False):
		tag_c=self.tag_c
		relevant_tags=self.tag[self.tag.value>3000]['tag']
		relevant_tags.values.remove_unused_categories(inplace=True)
		relevant_tag_index=relevant_tags.index.values
		relevant_tag_dic={i:t for i,t in zip(relevant_tag_index,relevant_tags)}

		if inplace:
			self.relevant_tags=relevant_tags
			self.relevant_tag_dic=relevant_tag_dic
			self.relevant_tag_index=relevant_tag_index
		else:
			return relevant_tags,relevant_tag_index,relevant_tag_dic
		

	
	def tag_prune(self,df):
    
		if 'tag' in df.columns:
			df=df[df.tag.isin(self.relevant_tags)]
			df.tag.values.remove_unused_categories(inplace=True)
		elif 'tag_index' in df.columns:
			df=df[df.tag_index.isin(self.relevant_tag_index)]
			      
		return df	
	


	
	
	
	def remove_dupplicates(self):
		#sometimes filings have duplicate figures against tags. It's a huge pain but difficult to find out why.
		
		tf=self.num.df.duplicated(['adsh','tag','ddate'],keep=False)

		print('There are',sum(tf), ' duplicated hetero values in the data which we will jettison')

		self.num.df=self.num.df[~tf]

	def create_feature_table_old(self,reorder=False):
		#####join num and sub dfs
		h=self.num.df
		filt=self.sub.df
		df=filt.merge(h,left_on=['adsh','period'],right_on=['adsh','ddate'],how='left')

		#join on previous period to allow difference to be taken between periods
		df=df.merge(h[['adsh','tag','ddate','value']],left_on=['adsh','prevper','tag'],right_on=['adsh','ddate','tag'],how='left',indicator='-1')
		df1=df[['cik','adsh','period','tag','value_x','value_y','form','filed']].dropna()
		df1['dvalue']=df1['value_x']-df1['value_y']

		#for convenience later on, rename current value column
		df1['value']=df1['value_x']

		#####organise data into a uniform sparse data table

		#Now to construct the period on period difference feature.

		#create a copy of the df
		#optionally I will restrict to static accounting fields
		ddf1=df1[df1.tag.isin(self.tag.df[self.tag.df.iord=="I"].tag)].copy

		#rename the tag field with a d
		ddf1['tag']=['d'+t for t in ddf1.tag]

		#create a value field with the difference value in it
		ddf1['value']=ddf1['dvalue']
		

		
		#create a copy of df
		pdf1=df1.copy()
		#rename tag field with a p
		pdf1['tag']=['p'+t for t in pdf1.tag]
		#create a value field with the previous period value in it
		pdf1['value']=pdf1['value_y']
		
		print(df1.shape)
		#append to the original df
		df1=df1.append(ddf1)
		print(df1.shape)
		
		#append to the original df
		df1=df1.append(pdf1)
		print(df1.shape)
		

		#do a group by to create a square data table
		df2=df1.groupby(['adsh','cik','period','tag'])['value'].last().unstack().reindex()
		self.FT=df2
		if reorder: self.reorder_cols()
		
		#get rid of infs
		self.FT.replace([np.inf, -np.inf], np.nan,inplace=True)
		
		#fill NANs with zeros
		self.FT.fillna(value=0,inplace=True)
	
	def create_feature_table(self,inplace=False,quarters=['0','4']):

		sub=self.sub.copy()
		num=self.num.copy()

		num=num[num.qtrs.isin(quarters)]
		num=self.convert_tag_index(num,columns=False)
		df1=num[['adsh','period','value','tag']].copy()
		df2=num[['adsh','period','value_p','tag']].copy()
		df2.rename(columns={'value_p':'value'},inplace=True)
		

		
		df2['tag']=['p'+t for t in df2.tag]
		
		ddf1=pd.concat([df1,df2],sort=True)
		
		ddf1=ddf1.groupby(['adsh','period','tag'])['value'].max().unstack().dropna(how='all')
		
		#sort columns nicely in descending order of data size
		ddf1=self.reorder_cols_by_data_count(df=ddf1)



		if inplace:
			self.FT=ddf1
	 
		else: return ddf1
		
	def convert_tag_index(self,df,dic=None,columns=True):
		if dic is None:
			dic=self.relevant_tag_dic
		if columns:	
			df=df.rename(columns=dic)
		else:
			df['tag']=df['tag_index'].map(dic)
		return df
		
		
	def reorder_cols(self):
		#just reorders columns nicely following creation of p and d tags
		c=['h']
		for k in self.FT.columns:
			if k[0] not in ['p','d']:
				if 'd'+k in self.FT.columns:
						   c.append([k,'p'+k,'d'+k])
		#this flattens the result
		c = [item for sublist in c[1:] for item in sublist]
		self.FT=self.FT[c]
		
	def reorder_cols_by_data_count(self,df=None,inplace=False):
		if df is None:
			df=self.FT
		columns=df.count().sort_values(ascending=False).index
		df=df[columns]
		
		if inplace:
			self.FT=df
		else:
			return df
		
	
	
	def normalise(self,cols,df2=None,inplace=False):
		if df2 is None:
			df2=self.FT
		#normalizes self.FT according to max value in cols. Drops records with no normalizer
		
		#need to get rid of records where there is no normalising value
			
		tf=np.all(df2[cols].isnull(),axis=1)
		df2=df2[~tf]
				
		print('Number of records dropped where no normaliser:',sum(tf))
		
		norms=df2[cols].max(axis=1).values
		df3=df2.div(norms, axis='rows')
		df3.replace([np.inf, -np.inf], np.nan,inplace=True)
		
		df3=self.reorder_cols_by_data_count(df=df3)
		
		if inplace:
			self.FT=df3
		else:
			return df3
	
	def threshhold_cut(self,thresh,df):
		#cuts out records with fewer than thresh non empty fields
		#leave df field empty to apply to self.FT
						
		df.fillna(value=0,inplace=True)
		prev_shape=df.shape
		df4=df[df.astype(bool).sum(axis=1)>thresh]
		new_shape=df4.shape
		print('Previous shape:',prev_shape,'New_shape:',new_shape, 'Data points lost:',prev_shape[0]-new_shape[0])
		return df4
			
	def plot_sparsity(self):
    
		fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)

		df3=self.FT.fillna(value=0)
		print(df3.shape)
		x=df3.astype(bool).sum(axis=1)
		x=x[x>0]

		mu=np.mean(x)
		sigma=np.sqrt(np.var(x))

		n, bins, patches = ax1.hist(x, 50, normed=0, facecolor='green', alpha=0.75)

		area = sum(np.diff(bins)*n)

		# add a 'best fit' line
		y = mlab.normpdf( bins, mu, sigma)
		l = ax1.plot(bins, area*y, 'r--', linewidth=1)

		tit=r'$\mathrm{Histogram\ of\ non empty field \#:}\ \mu='+ str(round(mu,2)) +',\ \sigma=' + str(round(sigma,2)) +'$'
		ax1.set(xlabel='#non empty fields',ylabel='#records',title=tit)

		#ax1.title(r'$\mathrm{Histogram\ of\ non empty field \#:}\ \mu='+ str(round(mu,2)) +',\ \sigma=' + str(round(sigma,2)) +'$')
		#plt.axis([40, 160, 0, 0.03])
		ax1.grid(True)

		cumulative=np.cumsum(n)
		ax2.bar(bins[:-1],cumulative,width=1.5)
		
	def boxplot_sec(self,df=None,ignore=['d','p']):
		#boxplot fo fields ignoring by default difference and previous fields
		if df is None: df=self.FT
		
		prop_cols=list(filter(lambda k: k[0] not in ignore, df.columns))
		axs,dic=df.boxplot(column=prop_cols,return_type='both',rot=90)
		
	def greedy_pruner(self,df,perc=99,threshhold=1):
		df.fillna(value=0,inplace=True)
		sdf=df
		df4=df
		big_cols=df4[df4.columns.values[df4[df4>5].count(axis=0)>50]].columns
		for k in sdf.columns:
			if k not in ['Assets','Liabilities','LiabilitiesAndStockholdersEquity'] and k[0] not in ['d','p'] and k not in big_cols:
				if np.percentile(sdf[k],perc)>0:
					sdf=sdf[sdf[k]<max(np.percentile(sdf[k],perc),threshhold)]
					print(k,sdf.shape)
			elif k in big_cols:
				sdf=sdf[sdf[k]<np.percentile(sdf[k],perc)]
				print('big',k,sdf.shape)
		return sdf  
	
	def fussy_pruner(self,df=None,perc=99):
		perc=perc/100
		if df is None: df=self.FT
		
		qu=df.quantile(perc)
		eek=[df.iloc[:,i]>qu[i] for i in range(len(qu))]
		ar=sum(pd.DataFrame(eek).values)
		tf=np.array(ar)
		x=[sum(tf<i) for i in range(0,20)]
		
		return tf,x
	
	def total_weight_prune(self,devs=2,dff=None,quant=False,qlb=0.15,qub=0.85):
    
		if dff is None:
			df=self.FT
			marker=True
		else:
			df=dff
			
		summary=df.sum(axis=1).describe()
		if quant:
			lb=df.sum(axis=1).quantile(qlb)
			ub=df.sum(axis=1).quantile(qub)
		else:
		
			lb=summary['mean']-devs*summary['std']
			ub=summary['mean']+devs*summary['std']
		ssdf=df[(df.sum(axis=1)>lb)&(df.sum(axis=1)<ub)]
		if dff is None:
			self.FT=ssdf
		else: return ssdf
	
		
class SecTable(object):
		
	def __init__(self,name=None,cfg=None,filters=None):
		self.name=name
		self.cfg=cfg
		import_d=SecLoader(self.cfg,name=name,mem_disp=True,filters=filters)
		self.df=import_d.df


if __name__ == "__main__":
	
	
	hom=os.path.normpath("C://Users//micro_zo50ceu//OneDrive - University College London//SEC//Dissertation_code//secTools")


	#this is the config file where I have stored the columns to import for each data source.
	loc=os.path.join(hom,"import_coeffs.yml")
	cfg=yamlLoad(loc)
