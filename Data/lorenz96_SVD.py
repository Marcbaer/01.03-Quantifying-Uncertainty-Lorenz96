#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:32:43 2018

@author: marcbar
"""
from __future__ import print_function
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as sk
import os # for saving
from matplotlib import rc
import pickle
import os
from sklearn.metrics import mean_squared_error
import sys
import argparse
import random as rand
import math
from scipy.integrate import ode


def SVD(F,step_factor=10000):
    
    base_path='./Simulation_Data/'
    
    
    with open(base_path + "/F"+str(F)+"_data_40.pickle", "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        data = pickle.load(file)
        X = data["X"]
        J = data["J"]
        dt = data["dt"]
        del data
            
    X_6=X
    # PCA
    
    X_mean_6 = np.mean(X_6,axis=0)
    X_mean_6 = np.reshape(X_mean_6, (1,-1))
    X_centered_6 = X_6 - np.matlib.repmat(X_mean_6, np.shape(X_6)[0], 1)
     
    step = np.shape(X_6)[0]//10000
    
    U_6, s_6, V_6 = np.linalg.svd(X_centered_6[::step,:])
    
    ## Energy spectrum
    Ek_6 = np.power(s_6,2)/np.sum(np.power(s_6,2))
    Ek_cum_6 = np.cumsum(Ek_6)
    
    #projection to eigenvalues
    
    c_6 = np.dot(X_centered_6, V_6)
    
    
    SVD6={'U_6':U_6, 's_6':s_6, 'V_6':V_6,'c_6':c_6}
    
    with open('./Data/F'+str(F)+'SVD6_results_40.pickle', "wb") as file:
        #Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(SVD6, file, protocol=2)
    #plot cumulative energy
        
    fig = plt.figure()
    #plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    K = np.shape(Ek_cum_6)[0]
    print(K)
    plt.plot(np.linspace(1, K, K), 100*Ek_cum_6, "-b", label="F=6")
    #plt.plot(np.linspace(1, K, K), 100*Ek_cum_16, "-r", label=r"$\nu=1/16$")
    plt.plot(np.linspace(1, K, K), 100*0.9*np.ones((K,1)), "--m", label=r"$90\%$ of total Energy")
    #plt.xscale("log")
    plt.xlim(1,K)
    plt.ylabel(r"Cummulative Energy in $\%$")
    plt.xlabel(r"Modes")
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig("./Figures/Plot_E_cum.pdf", bbox_inches="tight")
    plt.savefig("./Figures/Plot_E_cum.png", bbox_inches="tight")
    plt.plot()
    plt.close()
    
    return c_6,U_6,s_6,V_6

F=6
c_6,U_6,s_6,V_6=SVD(F)

with open('./Data/F'+str(F)+'c_6_40.pickle', "wb") as file:
#Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(c_6, file, protocol=2)
'''

plt.figure(figsize=(8,6))
plt.title('Singular values, F=6, J=40 ')
plt.plot(s_6,marker='o')
plt.xlabel('Mode')
plt.ylabel('Singular Value')
plt.legend()
plt.savefig('./Figures/Singular_Values.pdf')
plt.show() 

'''