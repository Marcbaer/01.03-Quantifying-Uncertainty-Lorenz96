'''
GP-LSTM regression on Lorenz96 data
'''

from __future__ import print_function
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

import pickle


class Lorenz96(object):
    def __init__(self,test=99,shift=1,pred_mode=1,input_modes=20,F=6,sequence_length=6,lr=1e-3,hdim=20,epochs=1500,batch_size=500):
        '''
        Initialize Lorenz96 Class

        Parameters
        ----------
        shift : Integer
            Number of steps to be predicted into the future.
        pred_mode : Integer
            Mode to be predicted into the future.
        input_modes : Integer
            Number of dimensions to be passed as information to the model for training.
        F : Integer
            Forcing Regime of the Lorenz96 system.
        test : Integer
            Experiment number.
        shift : Integer
            Number of steps to be predicted into the future.
        pred_mode : Integer
            Mode to be predicted into the future.
        input_modes : Integer
            Number of dimensions to be passed as information to the model for training.
        F : Integer
            Forcing Regime of the Lorenz96 system.
        '''
        #Data Parameters
        self.test=test
        self.shift=shift
        self.pred_mode=pred_mode
        self.input_modes=input_modes
        self.F=F
        self.sequence_length=sequence_length
        #Model and Training Parameters
        self.lr=lr
        self.hdim=hdim
        self.epochs=epochs
        self.batch_size=batch_size
        #load data
        self.load_data_lorenz()
        
        
    def load_data_lorenz(self):
        '''
        Load Lorenz 96 data and split into train, test and validation set.
        Define window size of input sequences.
        Assigns data to attribute self.data
        '''         
        total_length=self.sequence_length+self.shift
       
        #Load Data
        with open('./Data/40/F'+str(F)+'c_'+str(F)+'_40.pickle', "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            Data = pickle.load(file)
    
        data=Data[1000:,:self.input_modes]
        
        #create sequences with length sequence_length   
        result = []
        for index in range(len(data) - total_length):
            
            i=data[index: index + total_length]
            k=i[:self.sequence_length]
            k=np.append(k,i[total_length-1,:].reshape(1,self.input_modes),axis=0)
            result.append(k)
                       
        result = np.array(result) 
        #Train Test split
        train_end = int((90. / 100.) * len(result))
        
        result_train=result[:train_end]
        result_test=result[train_end:]
        
        #shuffle training data
        np.random.shuffle(result_train)
        
        #shape (#Timesteps,seq_length,#modes)    
        Input_data_train=result_train[:,:self.sequence_length,:]
        Output_data_train=result_train[:,-1,:]
    
        Input_data_test=result_test[:,:self.sequence_length,:]
        Output_data_test=result_test[:,-1,:]
        
        #Split sets
        train_end = int((80. / 100.) * len(result_train))
        
        X_train=Input_data_train[:train_end,:,:]
        y_train=Output_data_train[:train_end,:]
        
        X_test=Input_data_test[:,:,:]
        y_test=Output_data_test[:,:]
        
        X_valid=Input_data_train[train_end:,:,:]
        y_valid=Output_data_train[train_end:,:] 
      
        y_training=y_train[:,self.pred_mode-1]
        y_testing=y_test[:,self.pred_mode-1]
        y_validation=y_valid[:,self.pred_mode-1]
        
        #Reshape targets     
        y_training=y_training.reshape(y_training.shape[0],1)
        y_testing=y_testing.reshape(y_testing.shape[0],1)
        y_validation=y_validation.reshape(y_validation.shape[0],1)
        
        #standardize input sequences X
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
        
        self.data=data
    
    def build_train_GPLSTM(self):
        '''
        Define Model Architecture, load data and train the model.
    
        Returns
        -------
        history : Dictionnary
            Training Information.
        y_pred : Numpy Array
            Predicted output.
        var : Numpy Array
            Predicted Variances.
        rmse_predict : Float
            Training metrics.
        '''        
        data=self.data
        
        # Model & training parameters
        nb_train_samples = data['train'][0].shape[0]
        input_shape = data['train'][0].shape[1:]
        nb_outputs = len(data['train'][1])
        gp_input_shape = (1,)

        nn_params = {
            'H_dim': self.hdim,
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
        }
        
        # Retrieve model config
        nn_configs = load_NN_configs(filename='lstm.yaml',
                                     input_shape=input_shape,
                                     output_shape=gp_input_shape,
                                     params=nn_params)
        gp_configs = load_GP_configs(filename='gp.yaml',
                                     nb_outputs=nb_outputs,
                                     batch_size=self.batch_size,
                                     nb_train_samples=nb_train_samples,
                                     params=gp_params)
    
        # Construct & compile the model
        model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['MSGP']])
        loss = [gen_gp_loss(gp) for gp in model.output_layers]
        model.compile(optimizer=Adam(lr=lr), loss=loss)
    
        #learning rate scheduler
        callbacks = [EarlyStopping(monitor='val_mse', patience=2000)]
    
        # Train the model
        history = train(model, data, callbacks=callbacks, gp_n_iter=10,
                        checkpoint='checkp_shift_'+str(self.shift)+'_mode_'+str(self.pred_mode)+'_test_'+str(self.test), checkpoint_monitor='val_mse',
                        epochs=epochs, batch_size=self.batch_size, verbose=0)
        
        # Finetune the model
        model.finetune(*data['train'],
                       batch_size=self.batch_size,
                       gp_n_iter=100,
                       verbose=0)
          
        # Test the model
        X_test, y_test = data['test']
        X_train, y_train = data['train']
        
        y_pred,var = model.predict(X_test,return_var=True, X_tr=X_train, Y_tr=y_train,batch_size=self.batch_size)
        
        rmse_predict = RMSE(y_test, y_pred)
        print('Test predict RMSE:', rmse_predict)
        print('mean var:',np.array(var).mean())
        return history,y_test,y_pred,var,rmse_predict,model,data

#Train model and save results
if __name__ == '__main__':
    
    #Define Parameters
    F=6
    test=99
    shift=1
    sequence_length=6
    
    pred_mode=1
    input_modes=20
    lr=1e-5
    hdim=20
    epochs=1500
    batch_size=500
    
    #Call model
    Lorenz96=Lorenz96(test,shift,pred_mode,input_modes,F,sequence_length,lr,hdim,epochs,batch_size)
    history,y_test,y_pred,var,rmse_predict,model,data=Lorenz96.build_train_GPLSTM()
    
    y_test=np.array(y_test)
    y_pred=np.array(y_pred)
    
    std=[i**0.5 for i in var]
    std=np.array(std)
    std2=2*std
 
    validation_error=history.history['val_mse']
    training_error=history.history['mse']
    training_error=np.array(training_error)
    validation_error=np.array(validation_error)
    
    res={'y_test': y_test,'y_pred':y_pred,'var':var,'valid_error':validation_error,'train_error':training_error,'RMSE':rmse_predict,'std2':std2}
    pickle.dump(res, open('./Results/results_lorenz96_shift'+str(shift)+'_mode_'+str(pred_mode)+'_test_'+str(test)+'.p', "wb"))
