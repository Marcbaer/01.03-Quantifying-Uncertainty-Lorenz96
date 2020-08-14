#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:58:33 2018

@author: marcbar
"""
import numpy as np
from keras.models import load_model
from kgp.layers import GP
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
import matplotlib
#np.__version__
plt.rcParams["figure.figsize"] = [4., 3.]
SMALL_SIZE = 10
matplotlib.rc('axes', titlesize=SMALL_SIZE)
matplotlib.rc('font', size=SMALL_SIZE)
plt.tick_params(labelsize=10)


test=2
pred_mode=1

#2D
results = pickle.load(open('./Results/Results/res_propagation_test'+str(test)+'_predmode_'+str(pred_mode)+'.p', 'rb'))


#results = pickle.load(open('./Results/Results/res_propagation_2000_test9_predmode_1.p', 'rb'))

#1D
#results = pickle.load(open('./Results/Results/res_propagation1d_test'+str(test)+'.p', 'rb'))

#results = pickle.load(open('./old_results/Lorenz96_F6_exp1/results_lorenz3d_shift'+str(shift)+'_mode_'+str(mode)+'_test_'+str(test)+'.p', 'rb'))

K_d=results['K_d']
MEAN_d=results['MEAN_d']
VAR_d=results['VAR_d']
X_hist_d=results['X_hist_d']
W=results['W']
w=results['w']
mean_1=results['mean_1']
var_1=results['var_1']
y=np.array(results['y_test'])[0,:,0]

n_steps=len(MEAN_d)
n_samples=len(K_d['K1'])
#n_steps=results['n_steps']
#n_samples=results['n_samples']


#3D

point=results['point']
X=np.array(X_hist_d['hist0'][0])
X_initial=X[0,-1,0]
#X_initial=results['X_initial'][0,-1,0]


'''
#1D
point=2
X_initial=X_hist_d['hist0'][0][0,-1]
#X_initial=-3.2657751 
#plotting
'''
#plot initial starting point with errorbar
plt.scatter(w,X_initial,color='blue',label='sampled distribution')
#plt.scatter(1,mean_1,color='blue')
std1=var_1**0.5
std1=np.array(std1)
std1=3.5*std1
#plt.errorbar(1,mean_1,yerr=std1,capsize=0,fmt='',ecolor='lightgrey')

#plot sampled points
'''
#plot predicted means
plt.scatter(1,mean_1,color='blue')
plt.errorbar(1,mean_1,yerr=std1,capsize=0,fmt='',ecolor='lightgrey')
for i in range(2,n_steps-1):
    
    plt.scatter(W['w{0}'.format(i)],MEAN_d['mean{0}'.format(i+1)],color='deepskyblue')  
'''
#plot sampled points
#plot predicted means

for i in range(1,n_steps):
    pos=[i]
    violin=plt.violinplot(K_d['K{0}'.format(i)].reshape(len(K_d['K{0}'.format(i)])),pos,showmeans = True)
    for pc in [violin['cbars'],violin['cmins'],violin['cmaxes'],violin['cmeans']]:
        pc.set_edgecolor('#2222ff')
    for pc in violin['bodies']:
        pc.set_facecolor('#2222ff')
                      
plt.grid(False)
x=range(1,n_steps,1)
plt.plot(x,y[point:point+n_steps-1],color='red',marker='o',linestyle='--',label='target',fillstyle='none')
plt.xlabel('#step')
plt.title('Lorenz 96 propagation')

plt.legend(loc=1, prop={'size': 8})
plt.savefig('./Figures/L96_Prop_nsamples'+str(n_samples)+'_nsteps'+str(n_steps-1)+'.pdf')

plt.show()

'''WALK plot'''
x=[]
y1=[X_initial]
y2=[X_initial]
mean1=[]
for i in range(1,n_steps):
    pos=[i]
    y1.append(min(K_d['K{0}'.format(i)]))
    y2.append(max(K_d['K{0}'.format(i)]))
    x.append(i)
    mean1.append(K_d['K{0}'.format(i)].mean())
    
x1=np.append(0,x)
y1=np.array(y1).reshape(11,)
y2=np.array(y2).reshape(11,)    
plt.plot(x,mean1,label='predicted mean',color='blue',marker='o',linestyle='--')

plt.plot(x,y[point:point+n_steps-1],label='target',marker='o',linestyle='--',color='red',fillstyle='none')
plt.fill_between(x1,y1,y2,facecolor='lightgrey',label='confidence bound')
plt.scatter(0,X_initial,color='blue',label='initial point')
plt.xlabel('#step')
plt.title('Lorenz 96: Sampled distributions vs. true values')
plt.legend(loc=1, prop={'size': 8})
plt.savefig('./Figures/L96_uncertainty1.pdf')