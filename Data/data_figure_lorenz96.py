#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
from __future__ import print_function
import numpy as np
# Plotting parameters
import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm


from matplotlib import rc
from matplotlib  import cm
import matplotlib as mpl
#plt.rcParams["text.usetex"] = True

import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable


import pickle
import socket

#plt.style.use('default')
#plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='10'
plt.rcParams["figure.figsize"] = [5., 5.]
#matplotlib.rcParams.update({'font.size': 14})


hostname = socket.gethostname()

base_path='./Data/'

F = 10
j=40

#with open('./Data/40/F'+str(F)+'c_'+str(F)+'_40.pickle', "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
       # Data = pickle.load(file)

with open('./Data/40/F'+str(F)+'c_'+str(F)+'_40.pickle', "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        Data = pickle.load(file)
        X=Data
        
dt=0.01
J=40
X=Data

start=10000
N_plot=2500

x_plot = X[start:start+N_plot,:J]
N_plot = np.shape(x_plot)[0]
# Plotting the contour plot
#fig = plt.figure(figsize=(5,10))
t, s = np.meshgrid(np.arange(N_plot)*dt, np.array(range(J))+1)
plt.contourf(s, t, np.transpose(x_plot), 40, cmap=plt.get_cmap("seismic"))
plt.colorbar()
plt.title('F = '+str(F))
plt.xlabel(r"$Mode$")
plt.ylabel(r"$t$")
#font = {'weight':'normal', 'size':16}
#plt.rc('font', **font)


plt.savefig('./Figures/Plot_X_F'+str(F)+"_N"+str(N_plot)+"_40.png", dpi=100, bbox_inches="tight")

plt.show()

