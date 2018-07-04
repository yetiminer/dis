import numpy as np
import pandas as pd


def split_data_X_Y(df,cols):
    #for the

    Y_feature=cols

    tf=df[Y_feature]!=np.nan

    data=df.dropna(subset=Y_feature, how='any').copy()


    cols=data.columns
    cols_tf=~cols.isin(Y_feature)
    cols=cols[cols_tf]
    X=data[cols].values
    Y=data[Y_feature].values

    return X,Y

def remove_outlier(X,Y,level_x,level_y):
    print(Y.shape,'Y shape before threshhold cut')

    a=np.abs(Y)<2.5
    Y=Y[a]
    
    print(Y.shape, 'Y shape after cut')
    
    X=X[a.reshape(X.shape[0])]
    X[np.abs(X)>2.5]=np.nan

    return X,Y