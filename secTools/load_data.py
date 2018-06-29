import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from sklearn.model_selection import train_test_split
import pickle

def main():
	file_name='180611test'
	fileObject = open(file_name,'rb') 
	ds=pickle.load(fileObject)
	fileObject.close()
	
	h=ds.num.df
	filt=ds.sub.df
	df=filt.merge(h,left_on=['adsh','period'],right_on=['adsh','ddate'],how='left')
	df=df.merge(h[['adsh','tag','ddate','value']],left_on=['adsh','prevper','tag'],right_on=['adsh','ddate','tag'],how='left',indicator='-1')
	df0=df[['adsh','cik','period','tag','value_x']].copy()
	df1=df[['adsh','cik','period','tag','value_y']].copy()
	df0=df0.rename(columns={'value_x':'value'})
	df1=df1.rename(columns={'value_y':'value'})
	df1['tag']=['p'+str(t) for t in df1.tag]
	df0=df0.append(df1)
	df2=df0.groupby(['adsh','cik','period','tag'])['value'].last().unstack().reindex()
	
	
	new_order=[[j,'p'+j] for j in df2.columns[[k[0] not in ['p','d'] for k in df2.columns]]]
	new_order = [item for sublist in new_order for item in sublist]
	df2=df2[new_order]
	ds.FT=df2
	cols=['pAssets','pLiabilitiesAndStockholdersEquity']
	ds.normalise(cols)

	ds.remove_dupplicates()
	ds.FT.replace([np.inf, -np.inf], np.nan,inplace=True)
	ds.FT.fillna(value=0,inplace=True)
	
	quant_weight_df=ds.total_weight_prune(0.05,ds.FT,quant=True,qlb=0.2,qub=0.8)
	display(quant_weight_df.sum(axis=1).describe())
	data=quant_weight_df.values
	cols=quant_weight_df.columns
	
	return ds, data, cols

if __name__ == '__main__':
	ds, data, cols=main()