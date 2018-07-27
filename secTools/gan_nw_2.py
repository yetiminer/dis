from numpy.random import seed
seed(35)
from tensorflow import set_random_seed
set_random_seed(35)


from keras.initializers import Constant
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Lambda, Input, Dense, Masking, merge, Dropout,BatchNormalization,GaussianNoise
from keras.layers.merge import concatenate as concat
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model
from keras.utils import plot_model
from keras.initializers import glorot_normal
from keras.optimizers import Adam
from keras.losses import  binary_crossentropy

from custom_loss import sparse_recon_loss_mse
from gan_utils import make_trainable

def generator_nw(x_train,
				 g_noise=0.00,nodes=[64,16,64],y=False,prelu_bias=0.1,drop_ra=0.0,
				 ker_init=None,compile=True,output_dim=None):

	input_dim=x_train.shape[1]
	if output_dim is None:
		output_dim=input_dim
	
	fe1=nodes[0]
	Bottleneck_Dim=nodes[1]
	fe3=nodes[2]

	#Maybe 1 or 2 inputs:
	if y:
		X_in=Input(shape=(input_dim,),name='financial_cond_input')
		Y_in=Input(shape=(1,),name='financial_manip') #this is the dimension we are manipulating
		#concatenate the two inputs
		concat_en=concat([X_in,Y_in])
		inpu=[X_in,Y_in]
	else:
		X_in=Input(shape=(input_dim,),name='financial_cond_input')
		concat_en=X_in
		inpu=X_in

	#image encoder layers
	h1_en=Dropout(drop_ra,name='H1_dropout')(concat_en)
	h1_en=GaussianNoise(g_noise,name='H1_noise')(h1_en)
	print(fe1)
	h1_en=Dense(fe1,kernel_initializer=ker_init,name='H1_layer')(h1_en)
	h1_en=PReLU(name='H1_activation',alpha_initializer=Constant(value=prelu_bias))(h1_en)
	h1_en=BatchNormalization(name='H1_batch_norm')(h1_en)

	h2_en=Dropout(drop_ra,name='H2_dropout')(h1_en)
	h2_en=GaussianNoise(g_noise,name='H12_noise')(h2_en)
	h2_en=Dense(Bottleneck_Dim,kernel_initializer=ker_init,name='H2_layer')(h2_en)
	h2_en=PReLU(alpha_initializer=Constant(value=prelu_bias),name='H2_activation')(h2_en)
	Latent_space=BatchNormalization(name='H2_batch_norm')(h2_en)


	h3_dec=Dense(fe3,kernel_initializer=ker_init,name='H3_layer')(Latent_space)
	h3_dec=PReLU(alpha_initializer=Constant(value=prelu_bias),name='H3_activation')(h3_dec)

	out_dec=Dense(output_dim,name='Output_layer')(h3_dec)

	if y:
		out_dec=concat([out_dec,Y_in])



	Generator=Model(inpu,out_dec)
	Generator.summary()

	#if compile:
	#	gen_compile_dic={'loss':sparse_recon_loss_mse,'metrics':metrics,'optimizer':'adam','early_stop':ES}
	#	Generator.compile(**gen_compile_dic)

	return Generator
	
def discriminator_nw(x_train,g_noise=0.00,nodes=[64,16,64],y=False,prelu_bias=0.1,drop_ra=0.0,ker_init=None,compile=True):
	###Discriminator

	#Discriminator params
	n_x=x_train.shape[1]
	fd1=nodes[0]
	fd2=nodes[1]
	fd3=nodes[2]


	#image encoder layers
	X_candidate=Input(shape=(n_x+1,),name='X_RealorFake')

	h1_dis=Dropout(drop_ra,name='H1_dropout')(X_candidate)
	h1_dis=GaussianNoise(g_noise,name='H1_noise')(h1_dis)
	h1_dis=Dense(fd1)(h1_dis)
	h1_dis=PReLU(name='dH1_activation',alpha_initializer=Constant(value=prelu_bias))(h1_dis)

	h2_dis=Dropout(drop_ra,name='dH2_dropout')(h1_dis)
	h2_dis=GaussianNoise(g_noise,name='dH2_noise')(h2_dis)
	h2_dis=Dense(fd2)(h2_dis)
	h2_dis=PReLU(name='dH2_activation',alpha_initializer=Constant(value=prelu_bias))(h2_dis)

	out_dec=Dense(2,activation='softmax')(h2_dis)

	Discriminator=Model(inputs=X_candidate,outputs=out_dec)
	Discriminator.summary()

	#if compile:
	#	dis_compile_dic={'loss':binary_crossentropy,'optimizer':'adam'} #'early_stop':ES}
	#	Discriminator.compile(**dis_compile_dic)
		
	return Discriminator
	
def gan_nw(Generator,Discriminator,x_train):

	input_dim=x_train.shape[1]
	#build gan model
	make_trainable(Discriminator,False)
	gan_input=Input(shape=(input_dim,))
	gan_inputy=Input(shape=(1,))

	gan_gen_out=Generator([gan_input,gan_inputy])
	
	gan_output=Discriminator(gan_gen_out)

	GAN=Model([gan_input,gan_inputy],[gan_gen_out,gan_output])

	GAN.summary()
	
	#gan_loss_weights=[1E2,1]


	return GAN
	
def cond_gan_nw(Generator,Discriminator,x_train):

	input_dim=x_train.shape[1]
	#build gan model
	make_trainable(Discriminator,False)
	gan_input=Input(shape=(input_dim,))
	gan_inputy=Input(shape=(1,))
	
	gan_gen_out=Generator([gan_input,gan_inputy])
	gan_discrim_inp=concat([gan_gen_out,gan_input])
	
	gan_output=Discriminator(gan_discrim_inp)

	GAN=Model([gan_input,gan_inputy],[gan_gen_out,gan_output])

	GAN.summary()
	
	#gan_loss_weights=[1E2,1]


	return GAN
	
def generator_nw_5(x_train,
				 g_noise=0.00,nodes=[64,32,16,32,64],y=False,prelu_bias=0.1,drop_ra=0.0,
				 ker_init=None,compile=True,output_dim=None):

	input_dim=x_train.shape[1]
	if output_dim is None:
		output_dim=input_dim
	
	fe1=nodes[0]
	fe2=nodes[1]
	Bottleneck_Dim=nodes[2]
	fe3=nodes[3]
	fe4=nodes[4]

	#Maybe 1 or 2 inputs:
	if y:
		X_in=Input(shape=(input_dim,),name='financial_cond_input')
		Y_in=Input(shape=(1,),name='financial_manip') #this is the dimension we are manipulating
		#concatenate the two inputs
		concat_en=concat([X_in,Y_in])
		inpu=[X_in,Y_in]
	else:
		X_in=Input(shape=(input_dim,),name='financial_cond_input')
		concat_en=X_in
		inpu=X_in

	#image encoder layers
	h1_en=Dropout(drop_ra,name='H1_dropout')(concat_en)
	h1_en=GaussianNoise(g_noise,name='H1_noise')(h1_en)	
	h1_en=Dense(fe1,kernel_initializer=ker_init,name='H1_layer')(h1_en)
	h1_en=PReLU(name='H1_activation',alpha_initializer=Constant(value=prelu_bias))(h1_en)
	h1_en=BatchNormalization(name='H1_batch_norm')(h1_en)
	
	h2_en=Dropout(drop_ra,name='H2_dropout')(h1_en)
	h2_en=GaussianNoise(g_noise,name='H12_noise')(h2_en)
	h2_en=Dense(fe2,kernel_initializer=ker_init,name='H2_layer')(h2_en)
	h2_en=PReLU(alpha_initializer=Constant(value=prelu_bias),name='H2_activation')(h2_en)
	h2_en=BatchNormalization(name='H2_batch_norm')(h2_en)
	

	h3_en=Dropout(drop_ra,name='H3_dropout')(h2_en)
	h3_en=GaussianNoise(g_noise,name='H3_noise')(h3_en)
	h3_en=Dense(Bottleneck_Dim,kernel_initializer=ker_init,name='H3_layer')(h3_en)
	h3_en=PReLU(alpha_initializer=Constant(value=prelu_bias),name='H3_activation')(h3_en)
	Latent_space=BatchNormalization(name='H3_batch_norm')(h3_en)


	h4_dec=Dense(fe3,kernel_initializer=ker_init,name='H4_layer')(Latent_space)
	h4_dec=PReLU(alpha_initializer=Constant(value=prelu_bias),name='H4_activation')(h4_dec)

	h5_dec=Dense(fe4,kernel_initializer=ker_init,name='H5_layer')(h4_dec)
	h5_dec=PReLU(alpha_initializer=Constant(value=prelu_bias),name='H5_activation')(h5_dec)
	
	out_dec=Dense(output_dim,name='Output_layer')(h4_dec)

	if y:
		out_dec=concat([out_dec,Y_in])



	Generator=Model(inpu,out_dec)
	Generator.summary()

	#if compile:
	#	gen_compile_dic={'loss':sparse_recon_loss_mse,'metrics':metrics,'optimizer':'adam','early_stop':ES}
	#	Generator.compile(**gen_compile_dic)

	return Generator
	
def discriminator_nw_4(x_train,g_noise=0.00,nodes=[64,32,16,8],y=False,prelu_bias=0.1,drop_ra=0.0,ker_init=None,compile=True):
	###Discriminator

	#Discriminator params
	n_x=x_train.shape[1]
	fd1=nodes[0]
	fd2=nodes[1]
	fd3=nodes[2]
	fd3=nodes[3]

	#image encoder layers
	X_candidate=Input(shape=(n_x+1,),name='X_RealorFake')

	h1_dis=Dropout(drop_ra,name='H1_dropout')(X_candidate)
	h1_dis=GaussianNoise(g_noise,name='H1_noise')(h1_dis)
	h1_dis=Dense(fd1,name='dH1_layer')(h1_dis)
	h1_dis=PReLU(name='dH1_activation',alpha_initializer=Constant(value=prelu_bias))(h1_dis)

	h2_dis=Dropout(drop_ra,name='dH2_dropout')(h1_dis)
	h2_dis=GaussianNoise(g_noise,name='dH2_noise')(h2_dis)
	h2_dis=Dense(fd2,name='dH2_layer')(h2_dis)
	h2_dis=PReLU(name='dH2_activation',alpha_initializer=Constant(value=prelu_bias))(h2_dis)
	
	h3_dis=Dropout(drop_ra,name='dH3_dropout')(h2_dis)
	h3_dis=GaussianNoise(g_noise,name='dH3_noise')(h3_dis)
	h3_dis=Dense(fd3,name='dH3_layer')(h3_dis)
	h3_dis=PReLU(name='dH3_activation',alpha_initializer=Constant(value=prelu_bias))(h3_dis)

	out_dec=Dense(2,activation='softmax')(h3_dis)

	Discriminator=Model(inputs=X_candidate,outputs=out_dec)
	Discriminator.summary()

	#if compile:
	#	dis_compile_dic={'loss':binary_crossentropy,'optimizer':'adam'} #'early_stop':ES}
	#	Discriminator.compile(**dis_compile_dic)
		
	return Discriminator