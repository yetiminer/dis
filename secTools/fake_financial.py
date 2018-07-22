

from numpy.random import seed
seed(35)
from tensorflow import set_random_seed
set_random_seed(35)

import numpy as np
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import linear_model
from custom_loss import keras_eval 

def fake_financial(x_test,y_test,ds,reps,var_scale=1):
    #takes
    
    x_test_des=ds.FT.loc[y_test].describe()
    x_test_des.loc['std','pAssets']=0
    x_test_des.loc['std','pLiabilitiesAndStockholdersEquity']=0
    
    #generate a vector of probs that a field is populated
    prob_vec=(ds.FT.loc[y_test]>0).sum().values/x_test.shape[0]

    #get standard deviation of each field
    
    x_test_std=x_test_des.loc['std'].values
    
    def make_rand_alt(p):
        #generate a binary sequence length=#features, from this vector prob
        alteration_yes_no=np.random.binomial(1,prob_vec)

        #random vector with standard deviations according to variables
        random_alterations=np.random.normal(scale=var_scale*x_test_std)

        out=np.multiply(random_alterations,alteration_yes_no)
        
        return out
    
    final_out=list(map(make_rand_alt,range(1,reps+1)))

    return final_out

def make_fake_data(x_test,y_test,ds,num_fakes=7000,method='easy',var_scale=1):
    #adds noise to input financials. easy adds noise anywhere according to distribution of fields
    #and frequency of field. Hard method only adds noise to fields which are non-zero to begin with 
    
    alts=fake_financial(x_test,y_test,ds,num_fakes,var_scale=var_scale)
    if method=='easy':
        
        x_fake=x_test[0:num_fakes]+np.stack(alts)[:,0:200]
        x_fake[np.abs(x_fake>2.5)]=0
    elif method=='hard':
        
        x_fake=x_test[0:num_fakes]
        x_fake[x_fake>0]=x_fake[x_fake>0]+np.stack(alts)[:,0:200][x_fake>0]
        x_fake[np.abs(x_fake)>2.5]=0
    else:
        print("'method either 'easy' or 'hard'")
        raise AssertionError
    
    return x_fake


def evaluate_loss(x,y):
    losses=list(map(lambda x: np_sparse_loss(x_fit[x],x_test[x]),range(1,x_fit.shape[0])))
    losses=np.array(losses)
    return losses

def two_class_hist(losses,fake_losses):

    plt.figure()
    [n_re,bins,patches]=plt.hist(losses,bins=100,density=True,range=[0,1], alpha=0.7)
    [n_fa,bins,patches_f]=plt.hist(fake_losses,bins=bins,density=True,alpha=0.7)
    
    plt.legend(['Real','Fake'])
    plt.title('Distribution of Reconstruction Loss')
    return plt
	
def compare_fit_with_fake_data(AE,x_test,x_fake,loss=None):
   
	x_fit=AE.model.predict(x_test)
	x_fake_fit=AE.model.predict(x_fake)
	if loss is None:
		
		fake_losses=keras_eval(AE.loss,x_fake,x_fake_fit)
		losses=keras_eval(AE.loss,x_test,x_fit)
	else:

		fake_losses=loss(x_fake_fit,x_fake)
		losses=loss(x_test,x_fit)
	fig1=two_class_hist(losses,fake_losses)

	return fig1, losses, fake_losses	
	
	
def fit_model(losses,fake_losses,model=svm.LinearSVC):
    X=np.append(losses,fake_losses)
    Y=np.append(np.zeros(losses.shape),np.ones(fake_losses.shape))
    X=X.reshape(-1, 1)
    
    try:
        clf=model(random_state=35)
    except TypeError:
        clf=model()
    
    clf.fit(X,Y)
    type1_errors=sum(clf.predict(losses.reshape(-1,1)))
    print("type 1 errors: ", type1_errors)
    type2_errors=fake_losses.shape[0]-sum(clf.predict(fake_losses.reshape(-1,1)))
    print("type 2 errors: ",type2_errors)
    total_error_rate=(type1_errors+type2_errors)/X.shape[0]
    type1_error_rate=type1_errors/losses.shape[0]
    type2_error_rate=type2_errors/fake_losses.shape[0]
	
    return total_error_rate,type1_error_rate,type2_error_rate, clf