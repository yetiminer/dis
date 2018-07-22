
from keras.losses import mse, mean_squared_error, mean_absolute_error
from keras import backend as K
import numpy as np

def recon_loss_abs(y_true, y_pred):
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

    return mean_absolute_error(y_true*mask,y_pred*mask)

def recon_loss_mse(y_true, y_pred):
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

    return mse(y_true*mask,y_pred*mask)
	
def sparse_recon_loss_mse(y_true, y_pred):
	#this gets the numerator correct on the final denominator of the loss function
	
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    n=K.int_shape(y_true)[-1]
    ones=K.ones(shape=K.shape(y_true))
    denom=K.sum(mask*ones,axis=-1)
    
    return K.sum(K.square(y_true*mask-y_pred*mask),axis=-1)/denom

def make_sparse_recon_loss_var(sparse_recon_loss_combi):
	
	def sparse_recon_loss_var(y_true, y_pred):
		#calcs std dev of a vector of losses
		return K.std(sparse_recon_loss_combi(y_true,y_pred))
	
	return sparse_recon_loss_var


	
def sparse_recon_loss_abs(y_true,y_pred):
	#this gets the numerator correct on the final denominator of the loss function
	#does this make a difference? 
	mask_value=0

	mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
	ones=K.ones(shape=K.shape(y_true))
	n=K.sum(mask*ones,axis=-1)
	return 1/n*K.sum(K.abs(y_true*mask-y_pred*mask),axis=-1)
	

def recon_loss_hinge(y_true,y_pred):
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    y_true=K.sign(y_true*mask)
    y_pred=y_pred*mask
    
    return squared_hinge(y_true,y_pred)

def recon_loss_combi(y_true, y_pred):
    lamb=0.4
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

    return lamb*mse(y_true*mask,y_pred*mask)+(1-lamb)*mean_absolute_error(y_true*mask,y_pred*mask)
	
def make_recon_loss_combi(lamb):

    def recon_loss_combi(y_true, y_pred):

        mask_value=0
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

        return lamb*mse(y_true*mask,y_pred*mask)+(1-lamb)*mean_absolute_error(y_true*mask,y_pred*mask)
    
    return recon_loss_combi
	
def make_sparse_recon_loss_combi(lamb):

    def sparse_recon_loss_combi(y_true, y_pred):

        return lamb*sparse_recon_loss_mse(y_true,y_pred)+(1-lamb)*sparse_recon_loss_abs(y_true,y_pred)
    
    return sparse_recon_loss_combi
	
def keras_eval(function,var1,var2):
    keras_ans=K.eval(function(K.variable(var1), K.variable(var2)))
    return keras_ans


def make_loss_combi(lamb,loss1,loss2):

    def loss_combi(y_true, y_pred):

        return lamb*loss1(y_true,y_pred)+(1-lamb)*loss2(y_true,y_pred)
    
    return loss_combi	
	
	
#I was worried about testing the loss functions since tensor functions are not easy to evaluate.

def mk_np_sparse_loss(weight):
  	
	def np_sparse_loss(x_test,x_fit,weight=weight):
		tf=x_test==0
		x_fit[tf]=0

		#count the number of non zero entries
		denom=np.sum(~tf,axis=-1)


		l1=np.sum(np.abs(x_fit-x_test),axis=-1)/denom
		l2=np.sum(((x_fit-x_test)**2),axis=-1)/denom

		return l2*weight+(1-weight)*l1
	
	return np_sparse_loss

	
def test_sparse_function(x_test,x_fit,weight):
	
	sparse_recon_loss_combi=make_sparse_recon_loss_combi(weight)
	
	
	keras_ans=K.eval(sparse_recon_loss_combi(K.variable(x_test), K.variable(x_fit)))
	numpy_ans=np_sparse_loss(x_test,x_fit,weight)
	return keras_ans,numpy_ans
	