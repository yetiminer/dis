
from keras.losses import mse, mean_squared_error, mean_absolute_error
from keras import backend as K

def recon_loss_abs(y_true, y_pred):
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

    return mean_absolute_error(y_true*mask,y_pred*mask)

def recon_loss_mse(y_true, y_pred):
    mask_value=0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

    return mse(y_true*mask,y_pred*mask)


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