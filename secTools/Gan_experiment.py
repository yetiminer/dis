from numpy.random import seed
seed(35)
from tensorflow import set_random_seed
set_random_seed(35)

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.losses import  binary_crossentropy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

from custom_loss import (recon_loss_abs, make_recon_loss_combi,sparse_recon_loss_abs,
                         make_sparse_recon_loss_combi,
                         sparse_recon_loss_mse, make_sparse_recon_loss_var)

from Autoencoders import autoencoder

from general_loader import ds_from_db
from data_manipulation import remove_outlier, augment_x_linear,augment_x,split_data_X_Y,augment_x_df
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.initializers import glorot_normal

from gan_nw_2 import generator_nw,discriminator_nw,gan_nw
from gan_utils import Discrim_pre_train, train_for_n, plot_loss, train_for_n_mono

def GAN_nw_standard(augment=False,gen_pre_train=True,gen_weights=None,discrim_pre_train=True,mono_train=True,
					pre_train_dic=None,gen_layer_dic=None,dis_layer_dic=None,loss_weights=None,gan_train_dic=None):
	#gather the data
	ds_dic={'pickle_file':'ds180704'}
	ds=ds_from_db(**ds_dic)
	ds.normalise(['pAssets','pLiabilitiesAndStockholdersEquity'],reorder=True,inplace=True)
	ds.normalise(['pAssets','pLiabilitiesAndStockholdersEquity'],reorder=True,inplace=True)

	#split the revenue column out
	X,Y,idx,cols=split_data_X_Y(ds.FT.replace([np.inf, -np.inf], np.nan),['Revenues'])

	col_list=np.append(cols[0:199].values,'Revenues')
	X=X[:,0:199]

	#remove outliers
	X,Y,idx=remove_outlier(X,2.5,idx,Y=Y,y_thresh=False)
	XY=np.concatenate((X,Y),axis=1)

	#test train split
	amal_train,amal_test,idx_train,idx_test=train_test_split(XY,idx,test_size=0.25,random_state=30)
	x_train=amal_train[:,0:199]
	x_test=amal_test[:,0:199]
	y_test=amal_test[:,-1].reshape((x_test.shape[0],1))
	y_train=amal_train[:,-1].reshape((x_train.shape[0],1))

	if augment:
		#augment data
		cutoff=max(Y)
		df=ds.FT.loc[idx_train][col_list]
		x_train_aug=augment_x_df(np.concatenate((x_train,y_train),axis=1)
								 ,df,repeats=10,fit_col='Assets',seed=22,cutoff=cutoff,col_num=200)
								 
		x_train=x_train_aug[:,0:199]
		y_train=x_train_aug[:,-1].reshape((x_train.shape[0],1))				
	
	#define loss functions
	sparse_recon_loss_combi=make_sparse_recon_loss_combi(0.8)
	sparse_recon_loss_var=make_sparse_recon_loss_var(sparse_recon_loss_combi)

	loss=sparse_recon_loss_combi

	metrics=[sparse_recon_loss_mse,sparse_recon_loss_abs,sparse_recon_loss_var]

	
	ES=EarlyStopping(monitor='sparse_recon_loss_combi_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')

	#define general NW initiation parameters
	if pre_train_dic is None:
		train_dic={'epochs':100,'batch_size':128}			
		layer_p_dic={'drop_ra':0.1, 'g_noise':0.05,nodes:[64,16,64]}
	else:
		layer_p_dic=pre_train_dic['layer_dic']		
		train_dic=pre_train_dic['train_dic']
	
	ker_init=glorot_normal(seed=22)	
	layer_p_dic['ker_init']=ker_init
	
	#initalise generator network for as autoencoder for pretraining
	if gen_pre_train:
		pre_gen_compile_dic={'loss':loss,'metrics':metrics,'optimizer':Adam(lr=0.001)}
		pre_train_ae=generator_nw(x_train,**layer_p_dic,y=True)
		pre_train_ae.compile(**pre_gen_compile_dic)
		pre_x_train=np.concatenate((x_train,y_train),axis=1)
		pre_x_test=np.concatenate((x_test,y_test),axis=1)
		ES=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
		pre_train_ae.fit([x_train,y_train],pre_x_train,validation_data=([x_test,y_test], pre_x_test),epochs=50,batch_size=128,callbacks=[ES])	

		gen_weights=pre_train_ae.get_weights()

	#define genrator layer parameters
	if gen_layer_dic is None:
		gen_layer_dic=layer_p_dic
	
	
	#initialise generator
	gen_compile_dic={'loss':loss,'metrics':metrics,'optimizer':Adam(lr=0.001)}
	Generator=generator_nw(x_train,**gen_layer_dic,y=True)
	Generator.compile(**gen_compile_dic)
	
	if gen_weights is not None:
		Generator.set_weights(gen_weights)

	#intialise Discriminator NW
	if dis_layer_dic is None:
		dis_layer_dic={'drop_ra':0.0, 'g_noise':0.00,'nodes':[64,32,16]}		
	
	dis_layer_dic['ker_init']=ker_init
	
	dis_compile_dic={'loss':binary_crossentropy,'optimizer':Adam(lr=0.001)} #'early_stop':ES}
	
	Discrim=discriminator_nw(x_train,**dis_layer_dic)
	Discrim.compile(**dis_compile_dic)
	
	#Pre train Discriminator?
	if discrim_pre_train:
	
		XT_aug, y_hat=Discrim_pre_train(x_train,y_train,Discrim,train_size=1000)

	#initialise GAN network
	if loss_weights is None:	
		loss_weights=[5,1]
	
	
	gan_compile_dic={'loss':[loss,binary_crossentropy],'optimizer':Adam(lr=0.001),'loss_weights':loss_weights}
	GAN=gan_nw(Generator,Discrim,x_train)
	GAN.compile(**gan_compile_dic)
	
	if gan_train_dic is None:
		gan_train_dic={'nb_epoch':1000,'plt_frq':100,'batch_size':128,'test_size':64}
			
	train_dic={'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test,
			'Generator':Generator,
			'Discriminator':Discrim,'GAN':GAN,'plot':False}
		
	train_dic={**train_dic,**gan_train_dic}
	

	#train gan network
	if mono_train:
		results=train_for_n_mono(**train_dic)
	else:
		results=train_for_n(**train_dic)
		results['weight_hist']=np.array(loss_weights).reshape((1,2))

	#output pretty picture
	fig=plot_loss(results['losses'],**{'scale_control':[[0,3],[0,3],[0,2]],'loss_weights':results['weight_hist']})
	
	out_dic={'out_dic':results,'loss_fig':fig,'GAN':GAN,
	'Discrim':Discrim,'Generator':Generator,'pre_gen_weights':gen_weights}
	
	return out_dic