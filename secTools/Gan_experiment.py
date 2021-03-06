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
from gan_utils import Discrim_pre_train, train_for_n, plot_loss, train_for_n_mono

#import the networks
from gan_nw_2 import (generator_nw,discriminator_nw,gan_nw,generator_nw_5,discriminator_nw_3,cond_gan_nw,
						generator_nw_u,generator_nw_5_u)

from textwrap import wrap

def GAN_nw_standard(augment=False,gen_pre_train=True,gen_weights=None,discrim_pre_train=True,mono_train=True,
					pre_train_dic=None,gen_layer_dic=None,dis_layer_dic=None,loss_weights=None,gan_train_dic=None,
					generator_name='generator_nw',discriminator_name='discriminator_nw',GAN_name='gan_nw',
					weight_change=False,weight_change_dic=None):
	
	#attempt reproducibility
	from numpy.random import seed
	seed(35)
	from tensorflow import set_random_seed
	set_random_seed(35)
	
	
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
	
	print('Training set size ',x_train.shape[0])
	print('Test set size ',x_test.shape[0])
	
	#define loss functions
	sparse_recon_loss_combi=make_sparse_recon_loss_combi(0.8)
	sparse_recon_loss_var=make_sparse_recon_loss_var(sparse_recon_loss_combi)

	loss=sparse_recon_loss_combi
	metrics=[sparse_recon_loss_mse,sparse_recon_loss_abs,sparse_recon_loss_var]	
	ES=EarlyStopping(monitor='sparse_recon_loss_combi_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')

	#load the network types:
	nw_dic=define_network_dictionary()
	generator_nw=nw_dic['generator'][generator_name]
	discriminator_nw=nw_dic['discriminator'][discriminator_name]
	gan_nw=nw_dic['GAN'][GAN_name]
	
	#define genrator layer parameters
	if gen_layer_dic is None:
		gen_layer_dic={'drop_ra':0.1, 'g_noise':0.05,nodes:[64,16,64]}
	ker_init=glorot_normal(seed=22)
	gen_layer_dic['ker_init']=ker_init
	
	#define pretrain parameters
	if pre_train_dic is None:
		train_dic={'epochs':100,'batch_size':128}			
		

	layer_p_dic=gen_layer_dic
		
	
	
	#initalise generator network for as autoencoder for pretraining
	#ae_loss=np.zeros((1+len(metrics),1))
	ae_loss=np.full(1+len(metrics), np.nan)
	if gen_pre_train:
		pre_gen_compile_dic={'loss':loss,'metrics':metrics,'optimizer':Adam(lr=0.001)}
		pre_train_ae=generator_nw(x_train,**layer_p_dic,y=True)
		pre_train_ae.compile(**pre_gen_compile_dic)
		pre_x_train=np.concatenate((x_train,y_train),axis=1)
		pre_x_test=np.concatenate((x_test,y_test),axis=1)
		ES=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
		pre_train_ae.fit([x_train,y_train],pre_x_train,validation_data=([x_test,y_test], pre_x_test),**pre_train_dic,callbacks=[ES])	

		gen_weights=pre_train_ae.get_weights()
		ae_loss=pre_train_ae.evaluate(x=[x_test,y_test], y=pre_x_test)


	
	
	#initialise generator
	gen_compile_dic={'loss':loss,'metrics':metrics,'optimizer':Adam(lr=0.001)}
	Generator=generator_nw(x_train,**gen_layer_dic,y=True)
	Generator.compile(**gen_compile_dic)
	
	if gen_weights is not None:
		Generator.set_weights(gen_weights)

	#intialise Discriminator NW
	if dis_layer_dic is None:
		dis_layer_dic={'drop_ra':0.0, 'g_noise':0.00,'nodes':[64,32]}		
	
	dis_layer_dic['ker_init']=ker_init
	
	dis_compile_dic={'loss':binary_crossentropy,'metrics':['binary_accuracy'],'optimizer':Adam(lr=0.001)} #'early_stop':ES}
	
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
	
	#if weights are to be changed during training, the GAN should re-recompiled ocassionally
	if weight_change:
		del gan_compile_dic['loss_weights']
		print(gan_compile_dic)
		if weight_change_dic is None:
		#alpha and beta are the weights over training epochs.
		#compile freq is how often the gan is recompiled. Too often and training becomes very slow
			weight_change_dic={'weight_change':True,'alpha':np.linspace(loss_weights[0],10,gan_train_dic['nb_epoch']),
			'beta':np.ones(gan_train_dic['nb_epoch']),'compile_freq':100}
			
		weight_change_dic={**weight_change_dic,**{'gan_compile_dic':gan_compile_dic}}		
		train_dic={**train_dic,**gan_train_dic,**weight_change_dic}
		
	train_dic={**train_dic,**gan_train_dic}
	

	#train gan network
	if mono_train:
		results=train_for_n_mono(**train_dic,idx_test=idx_test)
	else:
		results=train_for_n(**train_dic,idx_test=idx_test)
		#results['weight_hist']=np.array(loss_weights).reshape((1,2))
	
	#add in the AE pre train loss
	results['AE_pre_train']=ae_loss	
		
	out_dic={'out_dic':results,'GAN':GAN,
	'Discrim':Discrim,'Generator':Generator,'pre_gen_weights':gen_weights}	
	#output pretty picture
	
	
	fig=plot_loss(results['losses'],**{'scale':[[0,3],[0,3],[0,2],[-0.1,1.1]],'loss_weights':results['weight_hist']})
	out_dic['loss_fig']=fig
	
	#create df og history
	out_dic['df']=create_df(out_dic)

	
	return out_dic



		
	
	
def define_network_dictionary():
	generator_nw_dic={'generator_nw':generator_nw,'generator_nw_5':generator_nw_5,
		'generator_nw_u':generator_nw_u,'generator_nw_5_u':generator_nw_5_u}
	discriminator_nw_dic={'discriminator_nw':discriminator_nw,'discriminator_nw_3':discriminator_nw_3}
	GAN_nw_dic={'gan_nw':gan_nw,'cond_gan_nw':cond_gan_nw}
	nw_dic={'generator':generator_nw_dic,'discriminator':discriminator_nw_dic,'GAN':GAN_nw_dic}
	return nw_dic
	
	
	
def create_df(out_dic):
    results=out_dic['out_dic']
    df_dic={}
    df_dic['GAN_loss_val']=pd.DataFrame(results['losses']['t_gan'],columns=['val Gan loss','val reconstruction','val detection error'])
    df_dic['Discrim_loss']=pd.DataFrame(results['losses']['d'],columns=['train binary cross entropy','train accuracy'])
    df_dic['GAN_loss']=pd.DataFrame(results['losses']['g'],columns=['GAN loss','reconstruction error','detection error'])
    df_dic['Discrim_acc_val']=pd.DataFrame(results['losses']['t_ac'],columns=['val accuracy','val real accuracy','val fake accuracy'])
    df_dic['Discrim_loss_val']=pd.DataFrame(results['losses']['t'],columns=['val binary cross entropy',
                                                              'val real binary cross entropy',
                                                              'val fake binary cross entropy'])
    df_dic['AE_loss']=pd.DataFrame(results['losses']['t_gen'],columns=['val AE Sparse loss','val AE mse','val AE abs','val AE var'])
    df_dic['Loss_weights']=pd.DataFrame(results['weight_hist'],columns=['Reconstruction_weight','BCE_weight'])
    df=pd.concat(df_dic,axis=1)
    return df
	
import uuid
import os
import yaml
from keras.utils import plot_model
from keras import Model

def save_results(out_dic,gan_dic,location=None):
    
	if location is None:
		location=os.getcwd()
		location=os.path.join(location,'gan_results')

	#create uid based on config file
	uid=uuid.uuid3(uuid.NAMESPACE_DNS,str(gan_dic))
	print(uid)
	new_folder='Exp_'+str(uid)
	new_folder=os.path.join(location,new_folder)

	os.makedirs(new_folder)
	folder_path=os.path.join(location,new_folder)

	#save config
	cfg_loc=os.path.join(folder_path,'expo_cfg_'+str(uid)+'.yaml')
	f = open(cfg_loc, "w")
	yaml.dump(gan_dic, f)
	f.close()

	#save figures
	for k,fig in out_dic.items():
		if isinstance(fig,plt.Figure):
			
			title=fig._suptitle.get_text()

				
			
			fig._suptitle.set_text("\n".join(wrap(title+ ' exp uID:' + str(uid), 60)))
			fig_loc=os.path.join(folder_path,k+str(uid)+'.png')
			fig.savefig(fig_loc)

	#save NW pics
	for nam,mod in out_dic.items():
		if isinstance(mod,Model):
			mod_path=os.path.join(folder_path,nam+'_pic_'+str(uid)+'.png')
			plot_model(mod, to_file=mod_path,show_shapes=True)

	#save history
	hist_loc=os.path.join(folder_path,'hist_'+str(uid)+'.yaml')
	f = open(hist_loc, "w")
	yaml.dump(out_dic['out_dic'], f)
	f.close()

	#save df
	df_loc=os.path.join(folder_path,'hist_df_'+str(uid)+'.csv')
	out_dic['df'].to_csv(df_loc)
	
	#create summary df
	record_num=str(uid)
	out_dic['final_results_df']=create_final_df(out_dic,record_num)
	record_df=gf(gan_dic,record_num).merge(pd.DataFrame.transpose(out_dic['final_results_df']),left_index=True,right_index=True)
	record_df.index.name='Exp'
	out_dic['record_df']=record_df
	
	df_loc=os.path.join(folder_path,'record_df_'+str(uid)+'.csv')
	out_dic['record_df'].to_csv(df_loc)
	
	return out_dic
	
	
def update_central_results(out_dic,central_location=None,name='central_results.csv'):
	if central_location is None:
		central_location=os.getcwd()
		central_location=os.path.join(central_location,'gan_results',name)
	
	record_df=out_dic['record_df']
	
	#load directory of results
	try:
		
		central_df=pd.read_csv(central_location,header=[0,1],index_col=0)
		
		#append new_results
	
		central_df=central_df.append(record_df)
	except FileNotFoundError:
		print('No existing result sumary table, creating new one')
		central_df=record_df
	
	#only using glorot normalisation here
	central_df.loc[:,('gen_layer_dic','ker_init')]='Glorot Normal'
	
	#save updated results ledger
	central_df.to_csv(central_location)
	return central_df
	
	
def create_final_df(out_dic,num):
    results=out_dic['out_dic']
    df_dic={}
    df_dic['GAN loss val']=pd.DataFrame(results['final_eval']['gan_test_loss'],
                                        index=['val Gan loss','val reconstruction','val detection error'])

    df_dic['Discrim acc val']=pd.DataFrame(results['final_eval']['test_loss_fake'],
                                          index=['val fake binary cross entropy','val fake accuracy'])
    df_dic['Discrim loss val']=pd.DataFrame(results['final_eval']['test_loss_real'],
                                            index=['val real binary cross entropy','val real accuracy'])
                                                            
    df_dic['AE loss']=pd.DataFrame(results['final_eval']['gen_test_loss'],
                                   index=['val AE Sparse loss','val AE mse','val AE abs','val AE var'])
    df_dic['Loss weights Final']=pd.DataFrame(results['weight_hist'][-1,:],
                                        index=['Reconstruction weight','BCE weight'])
    df_dic['AE PreTrain']=pd.DataFrame(results['AE_pre_train'],index=['val AE Sparse loss','val AE mse','val AE abs','val AE var'])
    
    df=pd.concat(df_dic,axis=0)
    df=df.rename(columns={0:num})
    return df



def gf(dic,num):
    new_dic=dic.copy()
    df_dic={}
    flat_dic={}
    
    for k,val in new_dic.items():

        if isinstance(val,dict):
            keys=[]
            for s,v in val.items():
                if isinstance(v,dict):
                    for ss,vv in v.items():
                        v[ss]=str(vv)
                    print(v)
                    df_dic[k+s]=pd.DataFrame(v,index=[num])
                    
                else:
                    keys.append(s)
                    if s in ['alpha','beta']:
                        try:
                            val[s]=v[-1]
                        except:
                            val[s]=str(v)
                            
                    else:
                        val[s]=str(v)
            
            f=dict((k, val[k]) for k in keys)
            df_dic[k]=pd.DataFrame(f ,index=[num])


        else:
            flat_dic[k]=str(val)

    df_dic['Parameters']=pd.DataFrame(flat_dic,index=[num])

    df=pd.concat(df_dic,axis=1)
    return df