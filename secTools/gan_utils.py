import matplotlib.pyplot as plt
from keras.losses import mse, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from IPython import display
from tqdm import tqdm
import numpy as np
import random

# Freeze weights in the discriminator training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
		
def plot_loss(losses,**kwargs):
		losses_temp=losses.copy()
		
		if 'loss_weights' in kwargs:
			loss_weights=kwargs['loss_weights']
		else:
			loss_weights=[1,1]
		
		try:
			for l in losses_temp:
				losses_temp[l]=np.array(losses[l])
		except ValueError:
			print(l)
			print(losses[l])
			
			raise

		#display.clear_output(wait=True)
		#display.display(plt.gcf())
		fig=plt.figure(figsize=(12,16))

		ax1=fig.add_subplot(411)
		#plt.figure(figsize=(12,8))
		ax1.plot(losses_temp["d"][:,0], label='discriminator loss')
		
		ax1.plot(losses_temp["g"][:,0], label='generator loss')
		ax1.plot(losses_temp["d"][:,1], label='discriminator accuracy')
		ax1.plot(losses_temp["g"][:,2], label='generator GAN_loss')
		
		ax1.legend()
		#ax1.set_yscale('log')        
		
		ax2=fig.add_subplot(412)
		ax2.plot(loss_weights[:,0]*losses_temp["g"][:,1], label='Reconstruction loss')
		ax2.plot(loss_weights[:,1]*losses_temp["g"][:,2], label='Detection loss')
		ax2.legend()
		
		
		ax3=fig.add_subplot(413)
		ax3.plot(losses_temp["t"][:,0], label='Test Discriminator loss')
		ax3.plot(losses_temp["t"][:,1], label='Test Discrim Real loss')
		ax3.plot(losses_temp["t"][:,2], label='Test Discrim Fake loss')
		ax3.legend()
		ax3.set_ylim(0,2)
		
		ax4=fig.add_subplot(414)
		ax4.plot(losses_temp["t_ac"][:,0], label='Test Discriminator Accuracy')
		ax4.plot(losses_temp["t_ac"][:,1], label='Test Discrim Real Accuracy')
		ax4.plot(losses_temp["t_ac"][:,2], label='Test Discrim Fake Accuracy')
		ax4.legend()
		ax4.set_ylim(0,2)
		
		if 'scale_control' in kwargs:
			scale=kwargs['scale_control']
			
			ax1.set_ylim(scale[0])
			ax2.set_ylim(scale[1])
			ax3.set_ylim(scale[2])
			ax4.set_ylim(scale[3])
		
		return fig

def recon_loss(y_true, y_pred):
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

    return mean_absolute_error(y_true*mask,y_pred*mask)

	
	
	

def make_batch(x_train,y_train,batch_size,real=True):
        
        
        # Make generative images
        trainidx = np.random.randint(0,x_train.shape[0],size=batch_size)
        real_image_batch_x = x_train[trainidx]
        real_image_batch_y=y_train[trainidx]
        
        
        trainidx = np.random.randint(0,x_train.shape[0],size=batch_size)
        gen_feed_batch_x=x_train[trainidx]
		##### here is where we ask the generator to exaggerate y coordinate.
        gen_feed_batch_y=np.multiply(y_train[trainidx],np.random.uniform(0.7,3,size=(batch_size,1))) 
        #,np.random.uniform(1,3,size=batch_size))
        if real:
            return gen_feed_batch_x, gen_feed_batch_y, real_image_batch_x,real_image_batch_y
        else:
            return gen_feed_batch_x, gen_feed_batch_y

			
def make_label_vector(batch_size,mono=False):
        y = np.zeros([2*batch_size,2])
        y[0:batch_size,1] = 1 #second column=1 implies image is genuine
        y[batch_size:,0] = 1
        
        if mono:
		#for when generator is being trained, want discriminator to label everything real
            y = np.zeros([batch_size,2])
            y[:,1] = 1 
        
        return y			
			
def make_batch_mono(x_train,y_train,batch_size,real=True,y_cond=None,rand_gen=None):
	
	#define default random generator
	if rand_gen is None:
		def rand_gen(batch_size):
			return np.random.uniform(0.7,3,size=(batch_size,1))
		

	# Make generative images
	trainidx = np.random.randint(0,x_train.shape[0],size=batch_size)
	real_image_batch_x = x_train[trainidx]
	real_image_batch_y=y_train[trainidx]

	if y_cond is not None: #return the conditional information if required
		real_image_batch_y_cond=y_cond[trainidx]
	else:
		real_image_batch_y_cond=None


	trainidx = np.random.randint(0,x_train.shape[0],size=batch_size)
	gen_feed_batch_x=x_train[trainidx]
	##### here is where we ask the generator to exaggerate y coordinate.
	gen_feed_batch_y=np.multiply(y_train[trainidx],rand_gen(batch_size)) 
	#,np.random.uniform(1,3,size=batch_size))
	if real:
		return real_image_batch_x,real_image_batch_y, real_image_batch_y_cond
	else:
		return gen_feed_batch_x, gen_feed_batch_y, real_image_batch_y_cond

			
			
def make_label_vector_mono(batch_size,real=True,real_smoothing=None):
	y = np.zeros([batch_size,2])

	if real:
	#for when generator is being trained, want discriminator to label everything real

		y[:,1] = 1 #second column=1 implies image is genuine
		if real_smoothing is not None:
			y[:,1]=np.random.binomial(1,real_smoothing,size=(batch_size))
		
	else:
		y = np.zeros([batch_size,2])
		y[:,0] = 1 #first column=1 for fakes

	return y			

		
def train_for_n(**kwargs):
	
	x_train=kwargs['x_train']
	y_train=kwargs['y_train']
	x_test=kwargs['x_test']
	y_test=kwargs['y_test']
	nb_epoch=kwargs['nb_epoch']
	plt_frq=kwargs['plt_frq']
	batch_size=kwargs['batch_size']
	test_size=kwargs['test_size']
	Generator=kwargs['Generator']
	Discriminator=kwargs['Discriminator']
	GAN=kwargs['GAN']
	plot=kwargs['plot']


	losses = {"d":[], "g":[],"t":[]} 

	for e in tqdm(range(nb_epoch)):  
		
		gen_feed_batch_x, gen_feed_batch_y, real_image_batch_x,real_image_batch_y=make_batch(x_train,y_train,batch_size)
		
		
		generated_images = Generator.predict([gen_feed_batch_x,gen_feed_batch_y])
		
		# Train discriminator on generated images
		real_image_concat=np.concatenate((real_image_batch_x,real_image_batch_y),axis=1)
		X = np.concatenate((real_image_concat, generated_images))
		y = make_label_vector(batch_size)
		#should shuffle here?
		
		make_trainable(Discriminator,True)
		d_loss  = Discriminator.train_on_batch(X,y)
		losses["d"].append(d_loss)
		

		# train Generator-Discriminator stack on input noise to non-generated output class
		gan_feed_batch_x, gan_feed_batch_y=make_batch(x_train,y_train,batch_size,real=False)               
		y2=make_label_vector(batch_size,mono=True)
		
		#freeze the coefficients on the discrim network for GAN/Generator training
		make_trainable(Discriminator,False)
		
		GAN_image_concat=np.concatenate((gan_feed_batch_x,gan_feed_batch_y),axis=1)
		g_loss = GAN.train_on_batch([gan_feed_batch_x,gan_feed_batch_y], [GAN_image_concat,y2] )
		losses["g"].append(g_loss)
		
		# Updates plots
		if plot and e%plt_frq==plt_frq-1:
			plot_loss(losses)
		
		################
		# Validation data
		Tgen_feed_x, Tgen_feed_y, Treal_image_x,Treal_image_y=make_batch(x_test,y_test,test_size)               
		Tgenerated_images = Generator.predict([Tgen_feed_x,Tgen_feed_y])        
		Ty = make_label_vector(test_size)
		Treal_image_concat=np.concatenate((Treal_image_x,Treal_image_y),axis=1)
		TX=np.concatenate((Treal_image_concat,Tgenerated_images))
		test_loss=Discriminator.evaluate(TX,Ty)
		
		
		
		test_loss_real=Discriminator.evaluate(Treal_image_concat,Ty[0:test_size,:])
		test_loss_fake=Discriminator.evaluate(Tgenerated_images,Ty[-test_size:,:])
		
		losses['t'].append([test_loss,test_loss_real,test_loss_fake])
		
	out_dic={}
	out_dic['losses']=losses
	
	return out_dic
		
def Discrim_pre_train(x_train,y_train,Discrim,train_size=1000,y_cond=None):

	#initate Discriminator network

	ntrain = train_size #choose initial training size for discriminator
	n=x_train.shape[0]

	#draw some financials at random
	trainidx = random.sample(range(0,n), ntrain)
	XT = x_train[trainidx]
	YT=y_train[trainidx]
	
	if y_cond is not None:
		YT_cond=y_cond[trainidx]
		XT_aug=np.concatenate((XT,YT,YT_cond),axis=1)
	
	else:
		XT_aug=np.concatenate((XT,YT),axis=1)


	#generate some 'fake' financials
	noise_gen = np.random.normal(0,1,size=[ntrain,XT_aug.shape[1]])
	noise_gen[XT_aug==0]=0 #make it similarly sparse

	#concatenate
	XT_aug_plus_noise = np.concatenate((XT_aug, noise_gen))
	print(XT_aug_plus_noise.shape)

	#make indicator variable
	y = np.zeros([2*ntrain,2])
	y[:ntrain,1]=1
	y[ntrain:,0]=1

	make_trainable(Discrim,True)
	Discrim.fit(XT_aug_plus_noise ,y,epochs=10,batch_size=32)
	y_hat=Discrim.predict(XT_aug_plus_noise)
	y_hat_idx = np.argmax(y_hat,axis=1)
	y_idx = np.argmax(y,axis=1)
	diff = y_idx-y_hat_idx
	n_tot = y.shape[0]
	n_rig = (diff==0).sum()
	acc = n_rig*100.0/n_tot
	print("Accuracy:",acc, "%", n_rig, ' of ', n_tot, 'correct')
	return XT_aug,y_hat		
	
def train_for_n_mono(**kwargs):
	
		
	x_train=kwargs['x_train']
	y_train=kwargs['y_train']
	
	if 'y_train_cond' in kwargs:
		y_train_cond=kwargs['y_train_cond']
	else:
		y_train_cond=None
	
	x_test=kwargs['x_test']
	y_test=kwargs['y_test']
	
	if 'y_test_cond' in kwargs:
		y_test_cond=kwargs['y_test_cond']
	else:
		y_test_cond=None
	
	real_smoothing=None	
	if 'real_smoothing' in kwargs:
		real_smoothing=kwargs['real_smoothing']

	
	nb_epoch=kwargs['nb_epoch']
	plt_frq=kwargs['plt_frq']
	
	batch_size=kwargs['batch_size']
	gen_batch_size=batch_size
	if 'gen_batch_size' in kwargs:
		gen_batch_size=kwargs['gen_batch_size']
	print(gen_batch_size)
	
	gen_epoch=1
	if 'gen_epoch' in kwargs:
		gen_epoch=kwargs['gen_epoch']
	
	test_size=kwargs['test_size']
	Generator=kwargs['Generator']
	Discriminator=kwargs['Discriminator']
	GAN=kwargs['GAN']
	plot=kwargs['plot']
	
	cond=False
	if 'cond' in kwargs:
		cond=kwargs['cond']
	
	#setup variables for results
	losses = {"d":[], "g":[],"t":[],'t_ac':[]} 
	out_dic={}
	out_dic['weight_hist']=[]
	
	compile_freq=100
	if 'weight_change' in kwargs:
	#if I want to dynamically change weights in the GAN I need to compile inside the function
		weight_change=kwargs['weight_change']
		
		alpha = kwargs['alpha']
		beta = kwargs['beta']
		gan_compile_dic=kwargs['gan_compile_dic']		
		GAN.compile(**gan_compile_dic,loss_weights=[alpha,beta])
		
		
		if compile_freq in kwargs:
			compile_freq=kwargs['compile_freq']
		
	else:
		weight_change=False
	
	rand_gen=None
	if 'rand_gen' in kwargs:
		rand_gen=kwargs['rand_gen']
	
	for e in tqdm(range(nb_epoch)):  
		
		#If I want to change weights I need to recompile, tried defining weights with tensors but didn't work
		if e%compile_freq==0 and weight_change: 
			GAN.compile(**gan_compile_dic,loss_weights=[alpha[e],beta[e]])
		
		
		#decide whether we ask to draw real or generated images - alternates on odd even epochs
		real=e%2==0
		if real:
			real_image_batch_x,real_image_batch_y,real_image_batch_y_cond=make_batch_mono(x_train,
				y_train,batch_size,real=real,y_cond=y_train_cond,rand_gen=rand_gen)
			
			if cond:
				
				X=np.concatenate((real_image_batch_y_cond,real_image_batch_y),axis=1)
			else:
				X=np.concatenate((real_image_batch_x,real_image_batch_y),axis=1)
			
			y = make_label_vector_mono(batch_size,real=real,real_smoothing=real_smoothing)
		else:
			gen_feed_batch_x,gen_feed_batch_y,_=make_batch_mono(x_train,
				y_train,batch_size,real=real,rand_gen=rand_gen)
			
			generated_images = Generator.predict([gen_feed_batch_x,gen_feed_batch_y])
			X=generated_images
			y = make_label_vector_mono(batch_size,real=real)
		
		# Train discriminator on generated images
				
		make_trainable(Discriminator,True)
		
		if cond: #when the GAN is conditional, give the discriminator the previous period financials
			
			if real:
				discrim_in=np.concatenate((X,real_image_batch_x),axis=1)
			else:
				discrim_in=np.concatenate((X,gen_feed_batch_x),axis=1)
			
			d_loss = Discriminator.train_on_batch(discrim_in,y)
		else: 
			d_loss  = Discriminator.train_on_batch(X,y)
		
		
		losses["d"].append(d_loss)
		
		#might want to train the generator for more than one epoch
		for k in range(0,gen_epoch):
			# train Generator-Discriminator stack on input noise to non-generated output class
			gan_feed_batch_x, gan_feed_batch_y,gan_feed_batch_y_cond=make_batch_mono(x_train,
				y_train,gen_batch_size,real=False,y_cond=y_train_cond,rand_gen=rand_gen)               
			#deliberately mislabel 
			y2=make_label_vector_mono(gen_batch_size,real=True)
			
			#freeze the coefficients on the discrim network for GAN/Generator training
			make_trainable(Discriminator,False)
			
			if cond: 
				GAN_image_concat=np.concatenate((gan_feed_batch_y_cond,gan_feed_batch_y),axis=1)
			else:
				GAN_image_concat=np.concatenate((gan_feed_batch_x,gan_feed_batch_y),axis=1)
				
			g_loss = GAN.train_on_batch([gan_feed_batch_x,gan_feed_batch_y], [GAN_image_concat,y2] )
		
		losses["g"].append(g_loss)
		
		# Updates plots
		if plot and e%plt_frq==plt_frq-1:
			plot_loss(losses)
		
		################
		# Validation data
		
		if cond:
			Tgen_feed_x, Tgen_feed_y, Tgen_feed_y_cond=make_batch_mono(x_test,
				y_test,test_size,real=False,y_cond=y_test_cond,rand_gen=rand_gen)
			
			Tgenerated_images = Generator.predict([Tgen_feed_x,Tgen_feed_y]) 
			
			Ty_fake=make_label_vector_mono(test_size,real=False)
			
			TX_fake=np.concatenate((Tgenerated_images,Tgen_feed_x),axis=1)
			test_loss_fake=Discriminator.evaluate(TX_fake,Ty_fake)
			
			Treal_image_x,Treal_image_y, Treal_image_y_cond=make_batch_mono(x_test,
				y_test,test_size,real=True,y_cond=y_test_cond,rand_gen=rand_gen)
			
			Ty_real=make_label_vector_mono(test_size,real=True)
			Treal_image_concat=np.concatenate((Treal_image_y_cond,Treal_image_y),axis=1)
			
			TX_real=np.concatenate((Treal_image_concat,Treal_image_x),axis=1)
			
			test_loss_real=Discriminator.evaluate(TX_real,Ty_real)
			
			test_loss=(test_loss_real+test_loss_fake)/2
			losses['t'].append([test_loss,test_loss_real,test_loss_fake])		
			
		else:

			
			#get a batch of real images, and target perturbations
			Tgen_feed_x, Tgen_feed_y,_=make_batch_mono(x_test,
				y_test,test_size,real=False,rand_gen=rand_gen)
			
			#put them through the generator
			TX_fake= Generator.predict([Tgen_feed_x,Tgen_feed_y]) 			
			
			#make correct labels
			Ty_fake=make_label_vector_mono(test_size,real=False)					
			
			#give to the discriminator
			test_loss_fake=Discriminator.evaluate(TX_fake,Ty_fake)
			
			#get a batch of real images
			TX_real,Treal_image_y,_=make_batch_mono(x_test,
				y_test,test_size,real=True,rand_gen=rand_gen)
				
			#make correct labels
			Ty_real=make_label_vector_mono(test_size,real=True)
			
			#feed to the discriminator
			Treal_image_concat=np.concatenate((TX_real,Treal_image_y),axis=1)
			test_loss_real=Discriminator.evaluate(Treal_image_concat,Ty_real)
			
			test_loss=(test_loss_real[0]+test_loss_fake[0])/2
			test_loss_ac=(test_loss_real[1]+test_loss_fake[1])/2
			
			losses['t'].append([test_loss,test_loss_real[0],test_loss_fake[0]])
			losses['t_ac'].append([test_loss_ac,test_loss_real[1],test_loss_fake[1]])
				
			out_dic['weight_hist'].append(GAN.loss_weights)
    
	out_dic['weight_hist']=np.array(out_dic['weight_hist'])
	out_dic['losses']=turn_into_array_dic(losses)
	
	return out_dic
	
def turn_into_array_dic(dic):
    od={}
    for k,li in dic.items():
        od[k]=np.array(li)
    return od
	
import uuid
import os
import yaml
from keras.utils import plot_model
from keras import Model

def save_results(out_dic,location=None):
    
    if location is None:
        location=os.getcwd()
    
    #create uid based on config file
    uid=uuid.uuid3(uuid.NAMESPACE_DNS,str(gan_dic))

    new_folder='Exp_'+str(uid)
    new_folder=os.path.join(location,new_folder)
    
    os.mkdir(new_folder)
    folder_path=os.path.join(location,new_folder)
    
    #save config
    cfg_loc=os.path.join(folder_path,'expo_cfg_'+str(uid)+'.yaml')
    f = open(cfg_loc, "w")
    yaml.dump(gan_dic, f)
    f.close()

    #save figure
    fig_loc=os.path.join(folder_path,'loss_fig_'+str(uid)+'.png')
    out_dic['loss_fig'].savefig(fig_loc)

    #save NW pics
    for nam,mod in out_dic.items():
        if isinstance(mod,Model):
            mod_path=os.path.join(folder_path,nam+'_pic_'+str(uid)+'.png')
            plot_model(mod, to_file=mod_path,show_shapes=True)

    #save history
    hist_loc=os.path.join(folder_path,'hist_'+str(uid)+'.yaml')
    f = open(cfg_loc, "w")
    yaml.dump(out_dic['out_dic'], f)
    f.close()