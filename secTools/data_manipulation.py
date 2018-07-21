import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from functools import reduce
from scipy.misc import comb

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

def remove_outlier(X,level_x,Y=None,level_y=None,replace_nan=True,replace_with=0):
    
	if Y is None:
	
		X[np.abs(X)>level_x]=np.nan
		
		if replace_nan:
			X[np.isnan(X)]=replace_with
			assert(~np.any(np.isnan(X)))
			
		return X
	
	else:
		print(Y.shape,'Y shape before threshhold cut')

		a=np.abs(Y)<level_y
		Y=Y[a]
		
		print(Y.shape, 'Y shape after cut')
		
		X=X[a.reshape(X.shape[0])]
		X[np.abs(X)>level_y]=np.nan
		
		if replace_nan:
			X[np.isnan(X)]=replace_with
			Y[np.isnan(Y)]=replace_with
			assert(~np.any(np.isnan(X)))
			assert(~np.any(np.isnan(Y)))

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
	x_train[np.abs(x_train)>2.5]=0
	return x_train_aug
	

def unique_pairs(arr):
    uview = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    uvals, uidx = np.unique(uview, return_inverse=True)
    pos = np.where(np.bincount(uidx) >= 2)[0]

    pairs = []
    for p in pos:
        pairs.append(np.where(uidx==p)[0])

    return np.array(pairs)
	
def new_datapoint(sample_pairs,x_train,pairs=False):    
	#pick set at random
	num_pairs=len(sample_pairs)

	if pairs:
	#so each possible pair has an equal chance of being picked
		ridx = np.random.choice(range(1,len(sample_pairs)), 1, [comb(len(k),2) for k in sample_pairs])[0]
	else:
		ridx=np.random.randint(1, high=num_pairs)
	match_set=sample_pairs[ridx]

	if pairs:
		num_match_set=2
		match_set_pair=np.random.choice(match_set,2,replace=False)
		weight_vec=sum_to_one_uniform(2)
		out=np.matmul(weight_vec,x_train[match_set_pair])
		out=np.reshape(out,(1,out.shape[0]))
		
	else:
	#new sample is linear combo of all in matching set
		num_match_set=len(match_set)
		weight_vec=sum_to_one_uniform(num_match_set)
		weight_vec=np.reshape(weight_vec,(1,weight_vec.shape[0]))

		out=np.matmul(weight_vec,x_train[match_set])

	out[np.abs(out)>2.5]=0
	return out
#add random linear combination of remaining financials 
#normalise

def sum_to_one_uniform(m):
    k=np.random.uniform(0,1,m)
    return k/sum(k)


	
def augment_x_linear(x_train,reps=20000,pairs=False):
	#randomly combine pairs or tuples of financials with the same set of missing values
	A=x_train!=0
	sample_pairs=unique_pairs(A)
	
	def new_dp_wrap(x):
		return new_datapoint(sample_pairs,x_train,pairs=pairs)
	
	num_financials=len(np.unique(reduce(lambda x,y: np.append(x,y),sample_pairs)))
	
	print("there are ",num_financials, " with at least one identical pair to compose new financials from")
	
	new_data=list(map(new_dp_wrap,range(0,reps)))
	
	new_data=np.vstack(new_data)
	
	return new_data
	

