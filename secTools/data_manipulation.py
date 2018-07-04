import numpy as np
import pandas as pd


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

def remove_outlier(X,Y,level_x,level_y):
    print(Y.shape,'Y shape before threshhold cut')

    a=np.abs(Y)<level_y
    Y=Y[a]
    
    print(Y.shape, 'Y shape after cut')
    
    X=X[a.reshape(X.shape[0])]
    X[np.abs(X)>level_y]=np.nan

    return X,Y