'''
GP-LSTM regression on Lorenz3D data
'''
from __future__ import print_function
import matlab.engine
import numpy as np

# Model assembling and executing
from Lorenz_prediction import Lorenz96
# Metrics & losses
from kgp.metrics import root_mean_squared_error as RMSE
#MPI and pickle imports
from mpi4py import MPI
import pickle


#initializes MPI
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

#set random seed
np.random.seed(42)

#Run Model training
if __name__ == '__main__':
    #set training parameters:
    test=13
    shift=1 
    pred_mode=1+rank
    
    input_modes=20
    F=6
    
    Lorenz96=Lorenz96(test=test,shift=shift,pred_mode=pred_mode,input_modes=input_modes,F=F)
    
    history,y_test,y_pred,var,rmse_predict,model,data=Lorenz96.build_train_GPLSTM()
    
    X_test, y_test = data['test']
    y_test=np.array(y_test)
    y_pred=np.array(y_pred)
    
    validation_error=history.history['val_mse']
    training_error=history.history['mse']
    training_error=np.array(training_error)
    validation_error=np.array(validation_error)
    
    res={'X_test':X_test,'y_test': y_test,'y_pred':y_pred,'var':var,'valid_error':validation_error,'train_error':training_error,'RMSE':rmse_predict}
    #pickle.dump(res, open('./Results/results_lorenz96_shift'+str(shift)+'_mode_'+str(pred_mode)+'_test_'+str(test)+'.p', "wb"))

global_state = np.zeros((input_modes))
local_state = np.zeros((input_modes))
local_state[rank] = rmse_predict

print('rank:',rank,'mean_var:',np.array(var)[0,:,0].mean(),'RMSE:',rmse_predict)

comm.Allreduce([local_state, MPI.DOUBLE], [global_state, MPI.DOUBLE], MPI.SUM)
if rank==0:print('global_state:',global_state)

#propagate uncertainty
    
X_test=data['test'][0]
X_train=data['train'][0]
y_train=data['train'][1]
y_test=np.array(data['test'][1])

#get initial point we want to propagate
point=250
X_initial=X_test[point:point+1,:,:]

#concatenated only to avoid dimension error
X1=np.concatenate((X_initial,X_initial),axis=0)

#predict mean&var for initial point
mean_1,var_1 = model.predict(X1,return_var=True,batch_size=500)

print('var',var_1)

xx=np.array(mean_1)
rmse=RMSE(y_test[0,point,0],mean_1[0][0])
print('rmse:',rmse)

mean_1=np.array(mean_1)[0,1,0]
var_1=np.array(var_1)[0,1,0]

#initialize dictionaries to track history
K_d={}
MEAN_d={}
VAR_d={}
X_hist_d={}

#append initial point to history
X_hist_d["hist{0}".format(0)]=[np.array(X_initial)]
MEAN_d["mean{0}".format(0)]=[mean_1]
VAR_d["var{0}".format(0)]=[var_1]

#number of steps propagating into the future
n_steps=10
# number of sampling points per step
n_samples=1000

for n in range(1,n_steps+1):
    K=[]
    MEAN=[]
    VAR=[]
    X_hist=[]
    for i in range(1,n_samples+1,1):
        
        #sample random integer to get index
        if n==1:
            k=0
        else:
            k=np.random.randint(n_samples)
            
        #get mean, var and history of sampled index
        u=MEAN_d["mean{0}".format(n-1)][k]
        v=VAR_d["var{0}".format(n-1)][k]
        std=v**0.5
        X_old=X_hist_d["hist{0}".format(n-1)][k]            
            
        #sample new point
        k2=np.random.normal(u,std)
        k2=np.array(k2).reshape(1,)
        K.append(k2)
    
        #fill global state with all sampled predictions of every mode
        global_state = np.zeros((input_modes))
        local_state = np.zeros((input_modes))
        local_state[rank] = k2
        comm.Allreduce([local_state, MPI.DOUBLE], [global_state, MPI.DOUBLE], MPI.SUM)
        
        #get new history and save it
        new_Hist=global_state
        new_Hist=new_Hist.reshape(1,1,input_modes)
        X_new=np.concatenate((X_old[:,1:,:],new_Hist),axis=1) 
              
        #Use new history to predict new mean and var
        X_=np.concatenate((X_old,X_new),axis=0)    
        mean_2,var_2=model.predict(X_,return_var=True)
        mean_2=np.array(mean_2)[0,1,0]
        var_2=np.array(var_2)[0,1,0]        
             
        #append new history to list 
        X_hist.append(X_new)
        MEAN.append(mean_2)
        VAR.append(var_2)        
	  
    #append new history to dictionnaries
    K_d["K{0}".format(n)]=np.array(K)
    MEAN_d["mean{0}".format(n)]=np.array(MEAN)
    VAR_d["var{0}".format(n)]=np.array(VAR)
    X_hist_d["hist{0}".format(n)]=np.array(X_hist)    
    print('step done:', n)
 
if rank==0:
    #define X-Axis points        
    W={}       
    for i in range(1,n_steps+1):
        
            W['w{0}'.format(i)]=np.full((n_samples),i)
            
    w=np.full((),0)  
    
    res={'point':point,'X_initial':X_initial,'K_d': K_d,'MEAN_d':MEAN_d,'VAR_d':VAR_d,'X_hist_d':X_hist_d,'W':W,'w':w,'mean_1':mean_1,'var_1':var_1,'n_samples':n_samples,'n_steps':n_steps,'y_test':y_test}

    pickle.dump(res, open('./Results/res_propagation_test'+str(test)+'_predmode_'+str(pred_mode)+'.p', "wb"))
    
    print('global_state:',global_state)
