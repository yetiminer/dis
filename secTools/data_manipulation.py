import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

def split_data_X_Y(df,cols):
	#for the

	Y_feature=cols

	#tf=df[Y_feature]!=np.nan
	#tf=~df[Y_feature].isin([0,np.nan]) #apparently there are some zero values floating about
	df[Y_feature]=df[Y_feature].replace([np.inf,0, -np.inf], np.nan)
	#df[Y_feature].replace([np.inf,-np.inf,0],np.nan,inplace=True)
	data=df.dropna(subset=Y_feature, how='any').copy()
	

	cols=data.columns
	cols_tf=~cols.isin(Y_feature)
	cols=cols[cols_tf]
	X=data[cols].values
	Y=data[Y_feature].values

	return X,Y, cols

def remove_outlier(X,level_x,Y=None,level_y=None):
    
	if Y is None:
	
		X[np.abs(X)>level_x]=np.nan
		
		return X
	
	else:
		print(Y.shape,'Y shape before threshhold cut')

		a=np.abs(Y)<level_y
		Y=Y[a]
		
		print(Y.shape, 'Y shape after cut')
		
		X=X[a.reshape(X.shape[0])]
		X[np.abs(X)>level_y]=np.nan

		return X,Y
		




def augment_x(x_train,ds,repeats=2,fit_col='Assets',seed=22):
    #augments data with random multiplactive constant on all present value columns
    #distribution of this constant is that of the eg Assets column (which is assets growth)
    
    
    aug_mask=list(map(lambda x: x[0]!='p',ds.FT.columns[0:200]))
    
    ker_fit_data=ds.FT[fit_col].values
    ker_fit_data=ker_fit_data[(ker_fit_data>0.5)*(ker_fit_data<1.5)]
    ker_fit_data=ker_fit_data.reshape(-1,1)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(ker_fit_data)
    x_train_aug=np.repeat(x_train,repeats,axis=0)
    
    number=x_train_aug.shape[0]
    seed=22
    scale_rand=kde.sample([number], seed)
    
    x_train_aug[:,aug_mask]*scale_rand
    return x_train_aug