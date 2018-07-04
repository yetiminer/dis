import matplotlib.pyplot as plt
from keras.losses import mse, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from IPython import display
from tqdm import tqdm

# Freeze weights in the discriminator training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
		
def plot_loss(losses,scale_control=True):
        losses_temp=losses.copy()
        try:
            for l in losses_temp:
                losses_temp[l]=np.array(losses[l])
        except ValueError:
            print(l)
            print(losses[l])
            
            raise
    
        display.clear_output(wait=True)
        display.display(plt.gcf())
        fig=plt.figure(figsize=(10,8))

        ax1=fig.add_subplot(311)
        #plt.figure(figsize=(10,8))
        ax1.plot(losses_temp["d"], label='discriminator loss')
        ax1.plot(losses_temp["g"][:,0], label='generator loss')
        ax1.legend()
        #ax1.set_yscale('log')        
        
        ax2=fig.add_subplot(312)
        ax2.plot(losses_temp["g"][:,1], label='L1_loss')
        ax2.plot(losses_temp["g"][:,2], label='Detection loss')
        ax2.legend()
        
        
        ax3=fig.add_subplot(313)
        ax3.plot(losses_temp["t"][:,0], label='Test Discriminator loss')
        ax3.plot(losses_temp["t"][:,1], label='Test Discrim Real loss')
        ax3.plot(losses_temp["t"][:,2], label='Test Discrim Fake loss')
        ax3.legend()
        ax3.set_ylim(0,2)
        
        if scale_control:
            ax1.set_ylim(0,10)
            ax2.set_ylim(0,10)
            ax3.set_ylim(0,10)
        
        plt.show()

def recon_loss(y_true, y_pred):
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

    return mean_absolute_error(y_true,y_pred)


def make_batch(x_train,y_train,BATCH_SIZE,real=True):
        
        
        # Make generative images
        trainidx = np.random.randint(0,x_train.shape[0],size=BATCH_SIZE)
        real_image_batch_x = x_train[trainidx]
        real_image_batch_y=y_train[trainidx]
        #real_image_concat=np.concatenate((real_image_batch_x,real_image_batch_y),axis=1)
        
        trainidx = np.random.randint(0,x_train.shape[0],size=BATCH_SIZE)
        gen_feed_batch_x=x_train[trainidx]
        gen_feed_batch_y=np.multiply(y_train[trainidx],np.random.uniform(1,3,size=(BATCH_SIZE,1))) ##### here is where we ask the generator to exaggerate y coordinate.
        #,np.random.uniform(1,3,size=BATCH_SIZE))
        if real:
            return gen_feed_batch_x, gen_feed_batch_y, real_image_batch_x,real_image_batch_y
        else:
            return gen_feed_batch_x, gen_feed_batch_y

def make_label_vector(BATCH_SIZE,mono=False):
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        if mono:
            y = np.zeros([BATCH_SIZE,2])
            y[:,1] = 1 
        
        return y	
		
def train_for_n(x_train,y_train,x_test,y_test,nb_epoch=100, plt_frq=25,BATCH_SIZE=32,test_size=32):
    losses = {"d":[], "g":[],"t":[]} 

    for e in tqdm(range(nb_epoch)):  
        
        gen_feed_batch_x, gen_feed_batch_y, real_image_batch_x,real_image_batch_y=make_batch(x_train,y_train,BATCH_SIZE)
        
        
        generated_images = Generator.predict([gen_feed_batch_x,gen_feed_batch_y])
        
        # Train discriminator on generated images
        real_image_concat=np.concatenate((real_image_batch_x,real_image_batch_y),axis=1)
        X = np.concatenate((real_image_concat, generated_images))
        y = make_label_vector(BATCH_SIZE)
        #should shuffle here?
        
        make_trainable(Discrim,True)
        d_loss  = Discrim.train_on_batch(X,y)
        losses["d"].append(d_loss)
        
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        gan_feed_batch_x, gan_feed_batch_y=make_batch(x_train,y_train,BATCH_SIZE,real=False)               
        y2=make_label_vector(BATCH_SIZE,mono=True)
        
        #freeze the coefficients on the discrim network for GAN/Generator training
        make_trainable(Discrim,False)
        
        GAN_image_concat=np.concatenate((gan_feed_batch_x,gan_feed_batch_y),axis=1)
        g_loss = GAN.train_on_batch([gan_feed_batch_x,gan_feed_batch_y], [GAN_image_concat,y2] )
        losses["g"].append(g_loss)
        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            plot_loss(losses)
        
        ################
        # Validation data
        Tgen_feed_x, Tgen_feed_y, Treal_image_x,Treal_image_y=make_batch(x_test,y_test,test_size)               
        Tgenerated_images = Generator.predict([Tgen_feed_x,Tgen_feed_y])        
        Ty = make_label_vector(test_size)
        Treal_image_concat=np.concatenate((Treal_image_x,Treal_image_y),axis=1)
        TX=np.concatenate((Treal_image_concat,Tgenerated_images))
        test_loss=Discrim.evaluate(TX,Ty)
        
        
        
        test_loss_real=Discrim.evaluate(Treal_image_concat,Ty[0:test_size,:])
        test_loss_fake=Discrim.evaluate(Tgenerated_images,Ty[-test_size:,:])
        
        losses['t'].append([test_loss,test_loss_real,test_loss_fake])
        
        return losses
		
		