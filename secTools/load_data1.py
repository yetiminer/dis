import os
import importlib
import yaml
import pytest
import pandas as pd
from secTools import importDataTools, SecLoader, yamlLoad
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab


class SECdataset(object):
	def __init__(self,name,cfg):
		self.name=name
		self.cfg=cfg
		self.sub=SecTable(name='sub',cfg=cfg[0])
		
		#define previous period
		self.sub.df['prevper']=self.sub.df.period.values.astype(int)-10000
		
		#filter out the financial companies'
		self.sub.df=self.sub.df[(self.sub.df.sic>=7000)|(self.sub.df.sic<6000)]
		
		#get the felevant filings codes
		self.relevant_filings=self.sub.df.adsh
		
		#create filter dictionary
		pre_filters={'adsh':self.relevant_filings}
		
		#load pre data
		self.pre=SecTable(name='pre',cfg=cfg[1],filters=pre_filters)

		#caluclate number of relevant filings
		relevant_filing_num=len(self.relevant_filings)
		tag_count=self.pre.df.groupby('tag')['adsh'].count()
		self.relevant_tags=tag_count[tag_count/relevant_filing_num>0.1].index
		
		#this is categorical so get rid of the unused categories
		self.relevant_tags=self.relevant_tags.remove_unused_categories()
		self.relevant_tags=self.relevant_tags.values.categories.tolist()
		print('number of tags appearing in 10% of filings:',len(self.relevant_tags),'expect 234')

		#create filter to import tag information. 
		#Abstract =0 gets rid of non numerical figures
		#data type is monetary gets rid of anything that isnt monetary - like shares
		tag_filters={'tag':self.relevant_tags,'abstract':[0],'datatype':['monetary']}
		#import tag information

		self.tag=SecTable(name='tag',cfg=cfg[2],filters=tag_filters)
		
		#recalculate relevant tag information
		self.relevant_tags=self.tag.df.tag.values.categories.tolist()
		print('there are', len(self.relevant_tags), ' tags left')
		#trim prez to only include relevant tags to save memory
		self.pre.df=self.pre.df[self.pre.df.tag.isin(self.relevant_tags)]

		#import num data
		num_filter={'tag':self.relevant_tags,'adsh':self.relevant_filings}
		self.num=SecTable(name='num',cfg=cfg[3],filters=num_filter)
		
		
		
	def remove_dupplicates(self):
		#sometimes filings have duplicate figures against tags. It's a huge pain but difficult to find out why.
		
		tf=self.num.df.duplicated(['adsh','tag','ddate'],keep=False)

		print('There are',sum(tf), ' duplicated hetero values in the data which we will jettison')

		self.num.df=self.num.df[~tf]

	def create_feature_table(self,reorder=False):
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
		
		
	
	def normalise(self,cols):
		#normalizes self.FT according to max value in cols. Drops records with no normalizer
		df2=self.FT
		#need to get rid of records where there is no normalising value
		tf=np.all(df2[cols]==0,axis=1)
		df2=df2[~tf]
		
		print('Number of records dropped where no normaliser:',sum(tf))
		
		norms=df2[cols].max(axis=1).values
		df3=df2.div(norms, axis='rows')
		self.FT=df3
	
	def threshhold_cut(self,thresh,df=None):
		#cuts out records with fewer than thresh non empty fields
		#leave df field empty to apply to self.FT
		if df is None:
			df=self.FT
			marker=True
		
				
		df.fillna(value=0,inplace=True)
		prev_shape=df.shape
		df4=df[df.astype(bool).sum(axis=1)>thresh]
		new_shape=df4.shape
		print('Previous shape:',prev_shape,'New_shape:',new_shape, 'Data points lost:',prev_shape[0]-new_shape[0])
		if df is None:
			self.FT=df4
		else:
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
		ax1.set(xlabel='#non empty fields',ylabel='#fields',title=tit)

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
