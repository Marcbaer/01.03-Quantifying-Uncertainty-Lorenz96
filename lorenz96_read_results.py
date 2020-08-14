#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:39:17 2018

@author: marcbar
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rcParams

plt.style.use('default')
SMALL_SIZE = 10
matplotlib.rc('axes', titlesize=SMALL_SIZE)
matplotlib.rc('font', size=SMALL_SIZE)
plt.tick_params(labelsize=10)
rcParams.update({'figure.autolayout': True})

#read results from workstation:
F=6
shift=1
mode=1
test=99


results=pickle.load(open('./Results/F6/Results/results_lorenz96_shift'+str(shift)+'_mode_'+str(mode)+'_test_'+str(test)+'.p', 'rb'))

#results=pickle.load(open('./Results/Results/results_l96_shift'+str(shift)+'_mode_'+str(mode)+'_test_'+str(test)+'.p', 'rb'))



y_test=results['y_test']
y_pred=results['y_pred']

rmse_predict=results['RMSE']
validation_error=results['valid_error']
training_error=results['train_error']
training_error=np.array(training_error)
training_error=training_error#/y_test.shape[0]

#edgecolor='grey'
#one step ahead
#var=1 std, to get 95% confidence (2 std in each direction), so var*2=95% confidence


var=results['var']
var=np.array(var)
#std
std1=[i**0.5 for i in var]

std1=np.array(std1)
std2=2*std1

#linestyle='dashed',markerfacecolor=None,markeredgecolor='blue',fillstyle='none'

size=500
point=1050
J=np.arange(0,y_test[0,size:size+size].shape[0],1)
plt.figure(figsize=(4,3))
plt.title('Predicted mean vs. true target, n= '+str(shift)+', RMSE=%1.2f' % rmse_predict)
plt.xlabel("#Test point")
#plt.ylabel('Value')
#plt.plot(J,y_pred[0,point:point+size,0], color='blue',label='predicted mean',linewidth=1.5)

#plt.fill_between(J,y_pred[0,point:point+size,0]+std2[0,point:point+size,0],y_pred[0,point:point+size,0]-std2[0,point:point+size,0],label='95% confidence',
    #alpha=0.8, facecolor='lightgrey')
plt.errorbar(J,y_pred[0,point:point+size,0],yerr=std2[0,point:point+size,0],capsize=0,fmt='',color='blue',ecolor='lightgrey',label='Predicted mean and 95% confidence')
   
plt.plot(y_test[0,point:point+size],label='true',color='red',linestyle='dashed')
plt.legend(loc=2, prop={'size': 6})
plt.savefig('./Figures/Lorenz96_Predictions_shift_'+str(shift)+'.pdf')
plt.show()


#Training convergence
plt.figure(figsize=(4,3))

plt.xlabel("# Epoch")
plt.ylabel('RMSE')
#plt.ylim(0,1)
plt.xlim(0,1000)
plt.title('Training convergence n='+str(shift))
plt.plot(validation_error,label='validation_error')
plt.plot(training_error,label='training_error')
plt.legend(loc=1, prop={'size': 8})
#plt.savefig('./Figures/Conv_F'+str(F)+'_shift'+str(shift)+'_mode'+str(mode)+'_test'+str(test)+'.pdf')
plt.show()

#plot estimated variances

k=abs(y_pred)-abs(y_test)
k=k[0,:,0]
i=len(k)
    
var=var[0,:,0]

print('min_var: ',min(var))
print('max_var:', max(var))  
  
plt.figure(figsize=(4,3))
plt.title('Variance histogram n='+str(shift))
plt.hist(var,bins=100,label='mean_variance='+str(round(var.mean(),6)))
plt.xlabel('variance')
plt.ylabel('#points')
plt.legend(loc=1, prop={'size': 8})
#plt.savefig('./Figures/Var_F'+str(F)+'_shift_'+str(shift)+'_mode_'+str(mode)+'_test_'+str(test)+'.pdf')
plt.show() 


'''
#step ahead pred
size=1000
J=np.arange(0,y_test[0,:size].shape[0],1)
plt.figure(figsize=(8,6))
plt.title('Predicted mean vs. true target, '+str(shift)+' step ahead,RMSE=%1.3f,F=%1.f' % (rmse_predict,F))
plt.xlabel("Test point")
plt.ylabel('Value')
plt.plot(y_test[0,:size],label='true',color='blue',linestyle='dashed',markerfacecolor=None,markeredgecolor='blue',fillstyle='none')
plt.plot(J,y_pred[0,:size,0], color='darkorange',label='predicted mean',linewidth=1.5)
plt.fill_between(J,y_pred[0,:size,0]+std2[0,:size,0],y_pred[0,:size,0]-std2[0,:size,0],label='95% confidence',
    alpha=0.8, facecolor='lightgrey')

plt.legend()
plt.savefig('./Figures/Lorenz3D_Predictions_shift_'+str(shift)+'.pdf')
plt.show()
'''

'''
#LSTM

size=50
J=np.arange(0,y_test[0,:size].shape[0],1)
plt.figure(figsize=(8,6))
plt.title('Predicted mean vs. true target, '+str(shift)+' step ahead,RMSE=%1.3f,F=%1.f' % (rmse_predict,F))
plt.xlabel("Test point")
plt.ylabel('Value')
plt.plot(y_test[0,:size],label='true',color='blue',linestyle='dashed',marker='o',markerfacecolor=None,markeredgecolor='blue',fillstyle='none')
plt.plot(y_pred[:size,0], color='darkorange',label='predicted mean',linewidth=1.5)

plt.legend()
plt.savefig('./Figures/Lorenz96_Predictions_shift_'+str(shift)+'_F'+str(F)+'.pdf')
plt.show()

#Training convergence
plt.figure(figsize=(8,6))
#plt.xlim((0,70))
plt.xlabel("# Epoch")
plt.ylabel('RMSE')

#plt.xlim(0,100)
plt.title('Lorenz96: Training convergence for '+str(shift)+' step ahead')
plt.plot(validation_error,label='validation_error')
plt.plot(training_error,label='training_error')
plt.legend()
plt.savefig('./Figures/TrainConv_shift'+str(shift)+'_mode_'+str(mode)+'_test_'+str(test)+'_F'+str(F)+'.pdf')
plt.show()
'''

'''
#plot variance propagation

fig = plt.figure()
var=[0.007,0.012,0.04,0.46]
x=[1,10,100,1000]
ax = plt.gca()
ax.scatter(x,var)
ax.set_yscale('log')
ax.set_xscale('log')
plt.title('Lorenz96: Variance Propagation')
plt.xlabel("shift")
plt.ylabel('Mean Variance')
plt.ylim(0.005,0.6)
plt.savefig('./Figures/L96_varporp.pdf')
plt.show()
'''
