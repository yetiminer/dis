
from keras.layers import (Lambda, Input, Dense, Masking, merge,
                          Dropout, Activation, GaussianNoise,
                          AlphaDropout, GaussianDropout,
                          BatchNormalization,activations, MaxoutDense)


from keras.layers.merge import concatenate as concat
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.models import Model, Sequential
from keras.initializers import Constant, Orthogonal
from keras.regularizers import l1,l2,l1_l2



import numpy as np
import pandas as pd
import os


def model_3h(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2,ker_init=None,nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_initializer=ker_init,
					bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						'noise':GaussianNoise(g_noise),'dropout_rate':drop_ra,'normalisation':BatchNormalization()
						   })

	layers.append({'layer':Dense(nodes[1],kernel_initializer=ker_init,
					bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						'noise':GaussianNoise(g_noise),'dropout_rate':drop_ra,'normalisation':BatchNormalization()
						   })


	layers.append({'layer':Dense(nodes[2],bias_initializer=Constant(value=bias),
							kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg),kernel_initializer=ker_init)})

	return layers



def model_5h(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2,ker_init=None,nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_initializer=ker_init,
					bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						'noise':GaussianNoise(g_noise)
						   })
	layers.append({'layer':Dense(nodes[1], kernel_initializer=ker_init,
							bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'noise':GaussianNoise(g_noise),
						   'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(nodes[2],bias_initializer=Constant(value=bias),kernel_initializer=ker_init),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias),kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias),
							kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg),kernel_initializer=ker_init)})

	return layers
	

def model_7h(X,drop_ra=0.1,l1_reg=0.001,bias=0.1,g_noise=0.2,ker_init=None,nodes=[32,16,8,4,8,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_initializer=ker_init,
					bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						'noise':GaussianNoise(g_noise)
						   })
	layers.append({'layer':Dense(nodes[1],input_dim=input_dim, kernel_initializer=ker_init,
							bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'noise':GaussianNoise(g_noise),
						   'dropout_rate':drop_ra,'normalisation':BatchNormalization()})
	
	layers.append({'layer':Dense(nodes[2],input_dim=input_dim, kernel_initializer=ker_init,
							bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'noise':GaussianNoise(g_noise),
						   'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias),kernel_initializer=ker_init),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[4],input_dim=input_dim,bias_initializer=Constant(value=bias),kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(nodes[5],input_dim=input_dim,bias_initializer=Constant(value=bias),kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})						   
						   
	layers.append({'layer':Dense(nodes[6],input_dim=input_dim,bias_initializer=Constant(value=bias),
							kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg),kernel_initializer=ker_init)})

	return layers