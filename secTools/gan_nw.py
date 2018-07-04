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
	
def discriminator_nw(**kwargs):
	###Discriminator
	
	#Discriminator params
	n_x=kwargs['input_dim']
	fd1=64
	fd2=32
	fd3=16
	ddout=0.2
	a=0.2 #alpha

	#image encoder layers
	X_candidate=Input(shape=(n_x+1,),name='X_RealorFake')

	h1_dis=Dense(fd1)(X_candidate)

	h2_dis=Dense(fd2)(h1_dis)
	h2_dis=LeakyReLU(alpha=a)(h2_dis)
	h2_dis=Dropout(ddout)(h2_dis)

	h3_dis=Dense(fd3)(h2_dis)
	h3_dis=LeakyReLU(alpha=a)(h3_dis)
	h3_dis=Dropout(ddout)(h3_dis)

	out_dec=Dense(2,activation='softmax')(h3_dis)
				
	Discriminator=Model(X_candidate,out_dec)
	Discriminator.summary()
	Discriminator.compile(loss='binary_crossentropy',optimizer='adam')
	return Discriminator
	
def gan_nw(Generator,Discriminator,**kwargs):
	n_x=kwargs['input_dim']
	gan_loss_weights=kwargs['gan_loss_weights']
	
	#build gan model
	make_trainable(Discriminator,False)
	gan_input=Input(shape=(n_x,))
	gan_inputy=Input(shape=(1,))

	gan_gen_out=Generator([gan_input,gan_inputy])
	gan_output=Discriminator(gan_gen_out)

	GAN=Model([gan_input,gan_inputy],[gan_gen_out,gan_output])

	GAN.summary()
	gan_loss=[recon_loss,K.binary_crossentropy]
	#gan_loss_weights=[1E2,1]

	GAN.compile(optimizer='adam',loss=gan_loss)
	return GAN
	
def generator_nw_unet(**kwargs):
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
	h1_en=Dropout(dout)(h1_en)
	
	h2_en=Dense(fe2)(h1_en)
	h2_en=LeakyReLU(alpha=a)(h2_en)
	h2_en=Dropout(dout)(h2_en)

	h3_en=Dense(fe3)(h2_en)
	h3_en=LeakyReLU(alpha=a)(h3_en)
	h3_en=Dropout(dout)(h3_en)

	h4_bot=Dense(BottleneckDim)(h3_en)
	h4_bot=LeakyReLU(alpha=a)(h4_bot)
	h4_bot=Dropout(dout)(h4_bot)
				
	h5_dec=Dense(fe5)(h4_bot)
	h5_dec=LeakyReLU(alpha=a)(h5_dec)
	h5_dec=Dropout(dout)(h5_dec)
	h5_dec=concat([h5_dec,h2_en])

	h6_dec=Dense(fe6)(h5_dec)
	h6_dec=LeakyReLU(alpha=a)(h6_dec)
	h6_dec=Dropout(dout)(h6_dec)
	h6_dec=concat([h6_dec,h1_en])
	
		
	out_dec=Dense(n_x)(h6_dec)
	out_dec=concat([out_dec,Y_in])
				
	Generator=Model([X_in,Y_in],out_dec)
	Generator.summary()
	Generator.compile(optimizer='adam',loss=recon_loss)
	return Generator