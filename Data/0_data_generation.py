#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
from scipy.integrate import ode
import numpy as np
import sys
import pickle
import socket
from lorenz96 import *


hostname = socket.gethostname()
'''
global_params = lambda:0
global_params.cluster = (hostname[:5]=='local') * 'local' + \
(hostname[:2]=='eu') * 'euler' + \
(hostname[:5]=='daint') * 'daint' + \
(hostname[:3]=='nid') * 'daint'

if global_params.cluster == 'euler':
    base_path = "/cluster/scratch/pvlachas/project_extreme_events/Code/Data/Lorenz96_F{:d}/Simulation_Data/"
elif global_params.cluster == 'local':
    base_path = "./Simulation_Data/"
elif global_params.cluster == 'daint':
    base_path = "/scratch/snx3000/pvlachas/project_extreme_events/Code/Data/Lorenz96_F{:d}/Simulation_Data/"
'''
base_path='/Users/marcbar/Documents/ETH/Master_project/Python/Applications/Lorenz96_create_data/Simulation_Data/'

base_path1='/cluster/home/marbaer/master_project/Lorenz96_F6/Simulation_Data'

# fix random seed for reproducibility
#np.random.seed(100)

# Forcing term
F = 8
J=40
print("# F="+str(F))
base_path.format(F)

T_transients = 1000
T = 1000
dt = 0.01 

N_transients = int(np.floor(T_transients/dt))
N = int(np.floor(T/dt))
print("Script to generate data for Lorenz96 for F={:d}".format(F))
print("Generating time-series with {:d} data-points".format(N))

# Data generation
X0 = F*np.ones((1,J))
X0 = X0 + 0.01 * np.random.randn(1,J)

# Initialization
X = np.matrix(np.zeros((N,J)))


print("Get past initial transients\n")
# Get past initial transients
for i in range(N_transients):
    X0 = RK4(Lorenz96,X0,0,dt, F);
    print("{:d}/{:d}".format(i, N_transients))
    sys.stdout.write("\033[F")

print("\n")
print("Generate time series\n")
# Generate time series
for i in range(N):
    X0 = RK4(Lorenz96,X0,0,dt, F);
    X[i,:] = X0
    print("{:d}/{:d}".format(i, N))
    sys.stdout.write("\033[F")


data = {
    "F":F,
    "J":J,
    "T":T,
    "T_transients":T_transients,
    "dt":dt,
    "N":N,
    "N_transients":N_transients,
    "X":X,
}

with open(base_path + "F"+str(F)+"_data_40.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    #pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    pickle.dump(data, file, protocol=2)


