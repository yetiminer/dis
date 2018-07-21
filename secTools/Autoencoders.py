from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from numpy.random import seed
seed(35)
from tensorflow import set_random_seed
set_random_seed(35)


from keras.layers import (Lambda, Input, Dense, Masking, merge,
                          Dropout, Activation, GaussianNoise,
                          AlphaDropout, GaussianDropout,
                          BatchNormalization,activations, MaxoutDense)


from keras.layers.merge import concatenate as concat
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.models import Model, Sequential, load_model
from keras.initializers import Constant, Orthogonal

from keras.utils import plot_model
from keras import backend as K
from keras.regularizers import l1,l2,l1_l2
from keras.callbacks import EarlyStopping

from custom_loss import (recon_loss_mse, recon_loss_abs,sparse_recon_loss_abs,
sparse_recon_loss_mse, make_recon_loss_combi,make_sparse_recon_loss_combi)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import yaml

#Build the sequential model

class autoencoder(object):
	def __init__(self,layers,name=None,train_data=[],test_data=[],
		batch_size=128,epochs=50,loss=sparse_recon_loss_mse, 
		metrics=[sparse_recon_loss_abs,sparse_recon_loss_mse],
		optimizer='adam',compile=True,early_stop=None):
		
		self.layers=layers
		self.name=name
		
		self.batch_size=batch_size
		self.epochs=epochs
		self.loss=loss
		self.train_data=train_data
		self.test_data=test_data
		self.metrics=metrics
		self.optimizer=optimizer
		
		self.model=self.build_model(layers=self.layers,inplace=False) #explicitly show that self.model is being set
		
		# define early stopping callback
		if early_stop is None:
			self.early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
		else:
			self.early_stop=early_stop

		#EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto', baseline=None)
		self.callbacks_list = [self.early_stop]
		
		if compile:
			self.compile()
			
	def compile(self):
		
		comp_dic={'loss':self.loss,'optimizer':self.optimizer}
		
		if self.metrics is not None:
			comp_dic['metrics']=self.metrics
		
		self.model.compile(**comp_dic)
		
	def fit(self):
		fit_dic={'batch_size':self.batch_size,'epochs':self.epochs,'validation_data':self.test_data}
		
		if self.callbacks_list is not None:
			fit_dic['callbacks']=self.callbacks_list
		
		x_train=self.train_data[0]
		y_train=self.train_data[1]
		
		self.History=self.model.fit(x_train,y_train,**fit_dic)
		

	#def fit_dictionary(self):
		#loads the parameters used in the standard fit function:
		
		# define early stopping callback
	#	earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=15, verbose=1, mode='auto')

		
	#	callbacks_list = [earlystop]
	#	batch_size=128
	#	epochs=120
	#	callbacks=callbacks_list
	#	validation_data=[x_test,x_test]
		


	def build_model(self,layers=None,inplace=True):
		#takes a list of layer dictionaries and returns a model
		model_h=Sequential()
		
		if layers is None:
			layers=self.layers
		
		for l in layers:
			model_h=self.layer_adder(model_h,**l)
		
		if inplace:
			self.model=model_h
		else:		
			return model_h


	def layer_adder(self, model_h,**kwargs):
		#adds layers to a model according to a dictionary of layer elements
		layer=kwargs['layer']
		model_h.add(layer)
		

			
		if 'noise' in kwargs:
			noise=kwargs['noise']
			model_h.add(noise)
			


		if 'normalisation' in kwargs:
			normalisation=kwargs['normalisation']
			model_h.add(normalisation)  
			


		if 'advanced_activation' in kwargs:
			adv_activation=kwargs['advanced_activation']
			model_h.add(adv_activation)	
			
		if 'activation' in kwargs:
			activation=kwargs['activation']
			model_h.add(Activation(activation))	
			
		if 'dropout_rate' in kwargs:
			dropout_rate=kwargs['dropout_rate']
			model_h.add(Dropout(dropout_rate))	
			
		
		return model_h

	def plot_loss(self,scale=None):
		fig=plt.figure(figsize=(10,8))
		ax1=fig.add_subplot(111)
		data_dic=self.History.history
		
		for key, val in data_dic.items():
			ax1.plot(np.log(val), label=key)
		
		ax1.legend()
		
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Log Loss')
		ax1.set_title('Training and Validation Loss for: '+ self.name)
		
		if scale is not None:
			
			ax1.set_xlim(scale[0])
			ax1.set_ylim(scale[1])

		self.LossPlot=fig	
			
		plt.show()	


def make_ae_dict_from_dict(layer_dic,layer_param_dic,model_data_dic,train_dic):
	#takes a dictionary of layers, layer params, model_data and training dict
	#builds and compiles autoencoder returning dictionary of AEs. All dictionaries
	#must share a key which is name of AEs
	def make_ae_model(model_layer,layer_param_dic,data_dic,name,train_dic):
		layers=model_layer(data_dic['train_data'][0],**layer_param_dic)

		#this command merges dictionaries into a new one without affecting old ones new as of python3.5
		new_train_dic={**data_dic,**train_dic}
		#print(new_train_dic)
		model=autoencoder(layers,name=name,**new_train_dic)
		return model

	ae_dict={}
	for k in layer_dic:
		ae_dict[k]=make_ae_model(layer_dic[k],layer_param_dic[k],model_data_dic[k],k,train_dic)
		
	return ae_dict

		
def model_node_wrap(nodes,ae_layers,x_train,layer_dic):
    #a function which makes a model a function of node_list only
    layers=ae_layers(x_train,nodes=nodes,**layer_dic)
    mod=autoencoder(layers,name=str(nodes),**train_dic)
    return mod
	

def model_loss_wrap(k,loss,model,train_dic,layer_dic,x_train):
    #a function which makes a model a function of loss only

    train_dic['loss']=loss
    layers=model(x_train,**layer_dic)
    mod=autoencoder(layers,**train_dic)
    return mod
	

def fit_model_dic(ae_dic):
    for mod in ae_dic:
        print(mod)
        ae_dic[mod].fit()	
		
def plot_loss_dic(ae_dic,val_loss='val_loss',loss='loss'):    
    fig=plt.figure(figsize=(15,15))
    ax1=fig.add_subplot(211)

    try:
        history_dic={key:val.model.history.history for key, val in ae_dic.items()}
    except AttributeError:
    #sometimes will just give this a dic of history arrays only
        history_dic=ae_dic

    for key, val in history_dic.items():

            ax1.plot(np.log(val[loss]), label=key)



    handles, labels = ax1.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax1.legend(handles, labels)

    ax1.set_title('Training Error')

    ax2=fig.add_subplot(212)
    for key, val in history_dic.items():

            ax2.plot(np.log(val[val_loss]), label=key)

    handles, labels = ax2.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax2.legend(handles, labels)

    ax2.set_title('Validation Error')

       
            
    return fig

def save_ae_dic(ae_dic,folder='ae_models/',path=None):
    if path is None:
        path=os.path.join(os.getcwd(),'ae_saves')
    
    file_path=os.path.join(path,folder)
    ensure_dir(file_path)
    
    for k,mod in ae_dic.items():

        save_loc=os.path.join(file_path,k+'.h5')
        
        mod.model.save(save_loc)
		
def save_ae_history(ae_dic,file_path):

	history_dic={key:val.model.history.history for key, val in ae_dic.items()}  
	ensure_dir(file_path)

	with open(file_path, 'w') as file_pi:
			yaml.dump(history_dic, file_pi)
	
        
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_to_keras_ae_dic(ae_dict):
    for k,val in ae_dict.items():
        is_keras=isinstance(val,Sequential)
        is_ae=isinstance(val,autoencoder)

    if is_keras:
        pass
    elif is_ae:
        ae_dic={k:val.model for k,val in ae_dict.items()}
    return ae_dic


def create_eval_dic(ae_dic,x_test,columns=['Loss','unweighted loss','mse loss','abs loss']):


    eval_dic={k:val.evaluate(x=x_test,y=x_test) for k,val in ae_dic.items()}


    df=pd.DataFrame.from_dict(eval_dic,orient='index',columns=columns)
    df.sort_values(by='Loss',axis=0,inplace=True)
    return df, eval_dic	

def create_params_dic(ae_dic):

    params_dic={}

    for k,mod in ae_dic.items():

        trainable_count = int(
            np.sum([K.count_params(p) for p in set(mod.trainable_weights)]))
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(mod.non_trainable_weights)]))

        params_dic[k]=[trainable_count,non_trainable_count]

       # print('Total params: {:,}'.format(trainable_count + non_trainable_count))
       # print('Trainable params: {:,}'.format(trainable_count))
       # print('Non-trainable params: {:,}'.format(non_trainable_count))
    df=pd.DataFrame.from_dict(params_dic,orient='index',columns=['Trainable params','Untrainable params'])
	
    return df, params_dic
	
def create_loss_param_table(ae_dic,x_test,columns=['Loss','unweighted loss','50-50 sparse loss','mse loss','abs loss']):
    #sometimes get passed dictionary of autoencoders not models
    ae_dic=convert_to_keras_ae_dic(ae_dic)
    
    df,eval_dic=create_eval_dic(ae_dic,x_test,columns)
    df2,param_dic=create_params_dic(ae_dic)
    df3=df.merge(df2,left_index=True,right_index=True)
    
    return df3
	

def load_ae_weights(ae_dic,folder='ae_archi_saves'):
    archi_dir=folder
    for key,mod in ae_dic.items():
        
        f=os.path.join(archi_dir,key+'.h5')
        mod.load_weights(f)	
		
def load_ae_dic(folder,custom_obj=None,combi_loss=0.5):
    
	if custom_obj is None:
		#keras needs to know about user defined object before loading model
		#since our loss functions are created with another function
		#they need to be defined first.....
		recon_loss_combi=make_recon_loss_combi(combi_loss)
		sparse_recon_loss_combi=make_sparse_recon_loss_combi(combi_loss)
		
		custom_obj={'sparse_recon_loss_combi': sparse_recon_loss_combi,'recon_loss_combi':recon_loss_combi,
			   'sparse_recon_loss_mse':sparse_recon_loss_mse,'sparse_recon_loss_abs':sparse_recon_loss_abs}

	
	file_list=os.listdir(folder)
	ae_dic={mod.split('.h5')[0]:load_model(os.path.join(folder,mod),custom_objects=custom_obj) for mod in file_list}
	return ae_dic

