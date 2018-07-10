from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import (Lambda, Input, Dense, Masking, merge,
                          Dropout, Activation, GaussianNoise,
                          AlphaDropout, GaussianDropout,
                          BatchNormalization,activations)


from keras.layers.merge import concatenate as concat
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.initializers import Constant, Orthogonal

from keras.utils import plot_model
from keras import backend as K
from keras.regularizers import l1,l2,l1_l2
from keras.callbacks import EarlyStopping

from custom_loss import recon_loss_mse, recon_loss_abs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

#Build the sequential model

class autoencoder(object):
	def __init__(self,layers,name=None,train_data=[],test_data=[],
		batch_size=128,epochs=50,loss=recon_loss_mse, 
		metrics=[recon_loss_abs,recon_loss_mse],
		optimizer='adam',compile=True):
		
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
		self.earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=15, verbose=1, mode='auto')

		#EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto', baseline=None)
		self.callbacks_list = [self.earlystop]
		
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
		

	def fit_dictionary(self):
		#loads the parameters used in the standard fit function:
		
		# define early stopping callback
		earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=15, verbose=1, mode='auto')

		#EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto', baseline=None)
		callbacks_list = [earlystop]
		batch_size=128
		epochs=120
		callbacks=callbacks_list
		validation_data=[x_test,x_test]
		


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
		

def model11(X,drop_ra=0.7,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	
	layers=[]

	layers.append({'layer':Dense(8,input_dim=input_dim,kernel_regularizer=l1(0.)),
						   'noise':GaussianNoise(g_noise)})
	layers.append({'layer':Dense(4,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'noise':GaussianDropout(drop_ra),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(2),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(4,input_dim=input_dim),
						   'advanced_activation':PReLU(),'noise':GaussianNoise(g_noise),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(8,input_dim=input_dim),
						   'advanced_activation':PReLU(),'noise':GaussianNoise(g_noise),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})
	
	return layers

def model111(X,drop_ra=0.7,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]


	layers=[]

	layers.append({'layer':Dense(8,input_dim=input_dim,kernel_regularizer=l1(0.)),
						   'noise':GaussianNoise(g_noise)})
	layers.append({'layer':Dense(4,input_dim=input_dim, bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'noise':GaussianDropout(drop_ra),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(3),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(4,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'noise':GaussianNoise(g_noise),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(8,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'noise':GaussianNoise(g_noise),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})
	
	return layers
	
def model1(X,drop_ra=0.7,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	
	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_regularizer=l1(0.)),
						   'noise':GaussianNoise(0.2)})
	layers.append({'layer':Dense(16,input_dim=input_dim, bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'noise':GaussianDropout(drop_ra),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(8,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'noise':GaussianNoise(g_noise),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'noise':GaussianNoise(g_noise),'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})
	
	return layers
	
def model_first(X,drop_ra=0.7,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_regularizer=l1(0.))
						   })
	layers.append({'layer':Dense(16,input_dim=input_dim, bias_initializer=Constant(value=bias),activation='relu'),
						   })

	layers.append({'layer':Dense(3,bias_initializer=Constant(value=bias),activation='relu'),
						   })

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias),activation='relu'),
						   })

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias),activation='relu'),
						   })

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})
	
	return layers
	
def model_second(X,drop_ra=0.7,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_regularizer=l1(0.))
						   })
	layers.append({'layer':Dense(16,input_dim=input_dim, bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(3,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})

	return layers
	
	
def model_third(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_regularizer=l1(0.))
						   })
	layers.append({'layer':Dense(16,input_dim=input_dim, bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'dropout_rate':drop_ra})

	layers.append({'layer':Dense(3,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})

	return layers
	
def model_fourth(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_regularizer=l1(0.))
						   })
	layers.append({'layer':Dense(16,input_dim=input_dim, bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(3,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})

	return layers
	
def model_fifth(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_regularizer=l1(0.),
							bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						   })
	layers.append({'layer':Dense(16,input_dim=input_dim, bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(3,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})

	return layers
	
	
def model_sixth(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_regularizer=l1(0.),
							bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						   })
	layers.append({'layer':Dense(16,input_dim=input_dim, bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(3,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg),bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	return layers
	
def model_seventh(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_regularizer=l1(0.),
					bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						'noise':GaussianNoise(g_noise)
						   })
	layers.append({'layer':Dense(16,input_dim=input_dim, bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'noise':GaussianNoise(g_noise),
						   'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(3,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   })

	return layers
	
def model_eighth(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2,ker_init=None):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(32,input_dim=input_dim,kernel_initializer=ker_init,
					bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						'noise':GaussianNoise(g_noise)
						   })
	layers.append({'layer':Dense(16,input_dim=input_dim, kernel_initializer=ker_init,
							bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'noise':GaussianNoise(g_noise),
						   'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(3,bias_initializer=Constant(value=bias),kernel_initializer=ker_init),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(16,input_dim=input_dim,bias_initializer=Constant(value=bias),kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(32,input_dim=input_dim,bias_initializer=Constant(value=bias),
							kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg),kernel_initializer=ker_init)})

	return layers