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
from keras.models import Model, Sequential
from keras.initializers import Constant, Orthogonal
from keras.regularizers import l1,l2,l1_l2



import numpy as np
import pandas as pd
import os




def model_first(X,drop_ra=0.,l1_reg=0.00,bias=0.1,g_noise=0.,nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements

	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_regularizer=l1(0.))
						   })
	layers.append({'layer':Dense(nodes[1], bias_initializer=Constant(value=bias),activation='relu'),
						   })

	layers.append({'layer':Dense(nodes[2],bias_initializer=Constant(value=bias),activation='relu'),
						   })

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias),activation='relu'),
						   })

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias),activation='relu'),
						   })

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})
	
	return layers
	
#try Prelu activation function	
def model_second(X,drop_ra=0.,l1_reg=0.00,bias=0.1,g_noise=0.,nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_regularizer=l1(0.)),
						   'advanced_activation':PReLU()})
	layers.append({'layer':Dense(nodes[1], bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[2],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})

	return layers
	
#add in dropout	functionality
def model_third(X,drop_ra=0.,l1_reg=0.,bias=0.1,g_noise=0.,nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_regularizer=l1(0.)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})
	layers.append({'layer':Dense(nodes[1], bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'dropout_rate':drop_ra})

	layers.append({'layer':Dense(nodes[2],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})

	return layers

#add in batch normalisation	
def model_fourth(X,drop_ra=0.,l1_reg=0.,bias=0.1,g_noise=0., nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_regularizer=l1(0.)),
						   'advanced_activation':PReLU(),'normalisation':BatchNormalization()})
	layers.append({'layer':Dense(nodes[1], bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(nodes[2],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})

	return layers
	
#try output prelu layer	
def model_sixth(X,drop_ra=0.,l1_reg=0.,bias=0.1,g_noise=0.,nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_regularizer=l1(0.),
							bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						   })
	layers.append({'layer':Dense(nodes[1], bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(nodes[2],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg),bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	return layers

#try gaussian noise layers in encoder	
def model_seventh(X,drop_ra=0.,l1_reg=0.,bias=0.1,g_noise=0.,nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_regularizer=l1(0.),
					bias_initializer=Constant(value=bias)), 'advanced_activation':PReLU(),
						'noise':GaussianNoise(g_noise)
						   })
	layers.append({'layer':Dense(nodes[1], bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(), 'noise':GaussianNoise(g_noise),
						   'dropout_rate':drop_ra,'normalisation':BatchNormalization()})

	layers.append({'layer':Dense(nodes[2],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU()})

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias)),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   })

	return layers

#try kernel initialisation	
def model_eighth(X,drop_ra=0.,l1_reg=0.,bias=0.1,g_noise=0.,ker_init=None,nodes=[32,16,5,16,32]):
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

#try ELU activation function	
def model_ninth(X,drop_ra=0.,l1_reg=0.,bias=0.,g_noise=0.,nodes=[32,16,5,16,32]):
	#a list of layers, each comprised of a dictionary of layer elements
	#K.clear_session()
	input_dim=X.shape[1]

	layers=[]

	layers.append({'layer':Dense(nodes[0],input_dim=input_dim,kernel_regularizer=l1(0.))
						   })
	layers.append({'layer':Dense(nodes[1], bias_initializer=Constant(value=bias)),
						   'advanced_activation':ELU()})

	layers.append({'layer':Dense(nodes[2]),
						   'advanced_activation':ELU()})

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias)),
						   'advanced_activation':ELU()})

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias)),
						   'advanced_activation':ELU()})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg)),
						   'activation':'linear'})

	return layers
	
def model_tenth(X,drop_ra=0.,l1_reg=0.,bias=0.,g_noise=0.,ker_init=None,nodes=[32,16,5,16,32]):
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

	layers.append({'layer':MaxoutDense(nodes[2],init=ker_init)
						   })

	layers.append({'layer':Dense(nodes[3],bias_initializer=Constant(value=bias),kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(nodes[4],bias_initializer=Constant(value=bias),
							kernel_initializer=ker_init),
						   'advanced_activation':PReLU(),'dropout_rate':drop_ra})

	layers.append({'layer':Dense(input_dim,kernel_regularizer=l1(l1_reg),kernel_initializer=ker_init)})

	return layers