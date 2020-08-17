from __future__ import print_function

'''
GP-LSTM regression on Lorenz3D data
'''
import matlab.engine
import numpy as np
# Keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
from kgp.utils.experiment import train
# Metrics & losses
from kgp.losses import gen_gp_loss
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

#set training parameters:
test=13
shift=1 
pred_mode=1+rank

#set dimensionality of input information
input_modes=20

#forcing regime of Lorenz96 system
F=6

#define functions

def load_data_lorenz(shift,pred_mode,input_modes,F):
    '''Function: Load Lorenz 96 data and split into train, test and validation set. Define window size of input sequences.
       Returns: Dictionnary containing train, test and validation DataFrames.
    '''
    sequence_length=12
    total_length=sequence_length+shift

    #Load Data
    with open('./Data/40/F'+str(F)+'c_'+str(F)+'_40.pickle', "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        Data = pickle.load(file)

    data=Data[:,:input_modes]
    
    #create sequences with length sequence_length  
    result = []
    for index in range(len(data) - total_length):
        
        i=data[index: index + total_length]
        k=i[:sequence_length]
        k=np.append(k,i[total_length-1,:].reshape(1,input_modes),axis=0)
        result.append(k)
              
    result = np.array(result) 
    
    #Train Test split
    train_end = int((90. / 100.) * len(result))
    
    result_train=result[:train_end]
    result_test=result[train_end:]
    
    #shuffle training data
    np.random.shuffle(result_train)
    
    #shape (#Timesteps,seq_length,#modes)    
    Input_data_train=result_train[:,:sequence_length,:]
    Output_data_train=result_train[:,-1,:]

    Input_data_test=result_test[:,:sequence_length,:]
    Output_data_test=result_test[:,-1,:]
    
    #Split sets
    train_end = int((80. / 100.) * len(result_train))
    
    X_train=Input_data_train[:train_end,:,:]
    y_train=Output_data_train[:train_end,:]
    
    X_test=Input_data_test[:,:,:]
    y_test=Output_data_test[:,:]
    
    X_valid=Input_data_train[train_end:,:,:]
    y_valid=Output_data_train[train_end:,:] 
  
    y_training=y_train[:,pred_mode-1]
    y_testing=y_test[:,pred_mode-1]
    y_validation=y_valid[:,pred_mode-1]
    
    #Reshape targets
    
    y_training=y_training.reshape(y_training.shape[0],1)
    y_testing=y_testing.reshape(y_testing.shape[0],1)
    y_validation=y_validation.reshape(y_validation.shape[0],1)
    
    #define output dictionnary
    data = {
        'train': [X_train, y_training],
        'valid': [X_valid, y_validation],
        'test': [X_test, y_testing],
    }
    
    # Re-format targets
    
    for set_name in data:
        y = data[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data[set_name][1] = [y[:,:,i] for i in range(y.shape[2])] 
    
    return data

def main(test,shift,pred_mode,input_modes,F):
    '''Function: Define Model Architecture, load data and train the model.
       Returns: Optimized Model and training outputs
    '''
    #Load data
    data=load_data_lorenz(shift,pred_mode,input_modes,F)
    
    # Model & training parameters
    nb_train_samples = data['train'][0].shape[0]
    input_shape = data['train'][0].shape[1:]
    nb_outputs = len(data['train'][1])
    gp_input_shape = (1,)
    batch_size = 500
    epochs = 1500

    nn_params = {
        'H_dim': 20,
        'H_activation': 'tanh',
        'dropout': 0.0,
    }

    gp_params = {
        'cov': 'SEiso', 
        'hyp_lik': np.log(0.1),
        'hyp_cov': [[3.0], [1.0]],
        
        'opt': {'cg_maxit': 10000,'cg_tol': 1e-4,
                'pred_var':-100,
                                
                },
        'grid_kwargs': {'eq': 1, 'k': 1000.},
        'update_grid': True,
        'ldB2_method':'lancz',
        'ldB2_lancz': True, 'ldB2_hutch':20,'ldB2_maxit':-50,

        'proj':'norm',     
    }
    
    # Retrieve model config
    nn_configs = load_NN_configs(filename='lstm.yaml',
                                 input_shape=input_shape,
                                 output_shape=gp_input_shape,
                                 params=nn_params)
    gp_configs = load_GP_configs(filename='gp.yaml',
                                 nb_outputs=nb_outputs,
                                 batch_size=batch_size,
                                 nb_train_samples=nb_train_samples,
                                 params=gp_params)

    # Construct & compile the model
    model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['MSGP']])
    loss = [gen_gp_loss(gp) for gp in model.output_layers]
    model.compile(optimizer=Adam(lr=1e-5), loss=loss)

    # Callbacks
    
    callbacks = [EarlyStopping(monitor='val_mse', patience=2000)]

    # Train the model
    history = train(model, data, callbacks=callbacks, gp_n_iter=10,
                    checkpoint='checkp96_shift_'+str(shift)+'_mode_'+str(pred_mode)+'_test_'+str(test), checkpoint_monitor='val_mse',
                    epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Finetune the model
    model.finetune(*data['train'],
                   batch_size=batch_size,
                   gp_n_iter=100,
                   verbose=0)
       
    # Test the model
    X_test, y_test = data['test']
    X_train, y_train = data['train']
    
    y_pred,var = model.predict(X_test,return_var=True, X_tr=X_train, Y_tr=y_train,batch_size=batch_size)
    
    rmse_predict = RMSE(y_test, y_pred)
    print('Test predict RMSE:', rmse_predict)
       
    return history,y_pred,var,rmse_predict,model,data

#Run Model training
if __name__ == '__main__':

    history,y_pred,var,rmse_predict,model,data=main(test,shift,pred_mode,input_modes,F)
    
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
