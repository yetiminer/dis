from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Masking, merge, Dropout

from keras.layers.merge import concatenate as concat
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.utils import plot_model
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

from gan_utils import make_trainable, recon_loss

#Build the sequential model
# define early stopping callback
earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=15, verbose=1, mode='auto')

#EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto', baseline=None)
callbacks_list = [earlystop]


def build_model(layers):

    model_h=Sequential()
    for l in layers:
        model_h=layer_adder(model_h,**l)
        
    return model_h


def layer_adder(model_h,**kwargs):
    
    layer=kwargs['layer']
    model_h.add(layer)
    
    if 'activation' in kwargs:
        activation=kwargs['activation']
        model_h.add(Activation(activation))
        
    if 'noise' in kwargs:
        noise=kwargs['noise']
        model_h.add(noise)
        
    if 'advanced_activation' in kwargs:
        adv_activation=kwargs['advanced_activation']
        model_h.add(adv_activation)

    if 'normalisation' in kwargs:
        normalisation=kwargs['normalisation']
        model_h.add(normalisation)  
        
    if 'dropout_rate' in kwargs:
        dropout_rate=kwargs['dropout_rate']
        model_h.add(Dropout(dropout_rate))

    
    return model_h

def autoencoder1(**kwargs):
K.clear_session()
input_dim=X.shape[1]
drop_ra=0.7
l1_reg=0.001

layers=[]

layers.append({'layer':Dense(8,input_dim=input_dim,kernel_regularizer=l1(0.)),
                       'noise':GaussianNoise(0.2)})
layers.append({'layer':Dense(4,input_dim=input_dim,),
                       'advanced_activation':PReLU(),'noise':GaussianDropout(drop_ra),'normalisation':BatchNormalization()})

layers.append({'layer':Dense(2),
                       'advanced_activation':PReLU()})

layers.append({'layer':Dense(4,input_dim=input_dim),
                       'advanced_activation':PReLU(),'noise':GaussianNoise(0.2),'normalisation':BatchNormalization()})

layers.append({'layer':Dense(8,input_dim=input_dim),
                       'advanced_activation':PReLU(),'noise':GaussianNoise(0.2),'normalisation':BatchNormalization()})

layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(0.001)),
                       'activation':'linear'})

def generator_nw(**kwargs):
	n_x=kwargs['input_dim']
	#first encoder
		
	#generator params

	
	fe1=64
	fe2=64
	fe3=32
	BottleneckDim=16
	fe5=32
	fe6=64
	a=0.2 #alpha
	dout=0.2 #dropout

	X_in=Input(shape=(n_x,),name='financial_cond_input')
	Y_in=Input(shape=(1,),name='financial_manip') #this is the dimension we are manipulating

	concat_en=concat([X_in,Y_in])

	#image encoder layers
	h1_en=Dense(fe1)(concat_en)

	h2_en=Dense(fe2)(h1_en)
	h2_en=LeakyReLU(alpha=a)(h2_en)
	h2_en=Dropout(dout)(h2_en)

	h3_en=Dense(fe3)(h2_en)
	h3_en=LeakyReLU(alpha=a)(h3_en)
	h3_en=Dropout(dout)(h3_en)

	h4_bot=Dense(BottleneckDim)(h3_en)
	h4_bot=LeakyReLU(alpha=a)(h4_bot)

				
	h5_dec=Dense(fe5)(h4_bot)
	h5_dec=LeakyReLU(alpha=a)(h5_dec)

	h6_dec=Dense(fe6)(h5_dec)
	h6_dec=LeakyReLU(alpha=a)(h6_dec)
				
	out_dec=Dense(n_x)(h6_dec)
	out_dec=concat([out_dec,Y_in])
				
	Generator=Model([X_in,Y_in],out_dec)
	Generator.summary()
	Generator.compile(optimizer='adam',loss=recon_loss)
	return Generator