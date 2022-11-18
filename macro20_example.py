#%%
import os
os.getcwd()
os.chdir('C:\\Users\\erica\\Dropbox\\_github\\sparse_varma')
os.getcwd()
#%%
import numpy as np
import pandas as pd
from numpy.random.mtrand import gamma
import torch
import scipy
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker as kron
from tensorly.tenalg import mode_dot
import itertools
from help_func import *
from DGP import *
from ALS import *
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(precision=3,suppress=True)
import networkx as nx
from matplotlib import pyplot as plot
import datetime,getopt
import csv

#%%
# macro_data_raw = pd.read_csv('macro20_new.csv', index_col=0)
# print(macro_data_raw.iloc[166,:])

#%%
macro_data = pd.read_csv('macro20_new.csv', index_col=0).T.to_numpy()
macro_data.shape

#%%
# start_train = 38
# end_train = 178
start_train = 0 # the value in R subtract 1
end_train = 166 # same as the value in R
N=20

P = 12 # lag for sparse var
lagr_mul_var = 0.008 # penalty for sparse var

macro_train = macro_data[:N, start_train:end_train]
T = end_train-start_train

vec_macro_train = np.ravel(macro_train[:,P:].reshape((-1,1)))
linear_matrix_1 = np.zeros((T-P,N*P))
for i in range(T-P):
    linear_matrix_1[i,:] = macro_train[:,range(i+P-1,i-1,-1)].T.reshape((1,-1))
linear_matrix = np.kron(np.eye(N), linear_matrix_1)

lasso = linear_model.Lasso(alpha=lagr_mul_var, max_iter = 1000)
lasso.fit(linear_matrix, vec_macro_train)
lasso_coef = lasso.coef_

A_ini = torch.from_numpy(lasso_coef.reshape((N,P,N))).permute((0,2,1))

#%%
print(sum(lasso_coef!=0))

for i in range(P):
    print(np.linalg.norm(A_ini[:,:,i], 'fro'))

#%%
P_est = P
s = 0
max_iter = 80
lr_omega = 0.02 # step size 0.030, 0.025
gamma = []; theta = [] # gamma = [-0.4]; theta = [1.05]

stop_thres = 1e-2
stop_method = 'SepEst'

### The following parameters can be searched over a range:
p=1
lmbd = [-0.2]
r = len(lmbd)
# lmbd_range = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
lagr_mul = [0.16] # tuning parameter for g, -0.18
# lagr_mul_range = [0.25, 0.3, 0.35]

forecast_len = 194 - end_train

FE1 = pd.DataFrame()
FE2 = pd.DataFrame()

y_predict_h = np.zeros((N, forecast_len))

prid_h_err1 = [] # abs error
prid_h_err2 = [] # squ error

flags = []

for k in range(forecast_len):
    print(k)
    T_test = end_train+k
    
    macro_train = macro_data[:N, start_train:T_test]
    T = T_test-start_train

    vec_macro_train = np.ravel(macro_train[:,P:].reshape((-1,1)))
    linear_matrix_1 = np.zeros((T-P,N*P))
    for i in range(T-P):
        linear_matrix_1[i,:] = macro_train[:,range(i+P-1,i-1,-1)].T.reshape((1,-1))
    linear_matrix = np.kron(np.eye(N), linear_matrix_1)

    lasso = linear_model.Lasso(alpha=lagr_mul_var, max_iter = 1000)
    lasso.fit(linear_matrix, vec_macro_train)
    lasso_coef = lasso.coef_

    A_ini = torch.from_numpy(lasso_coef.reshape((N,P,N))).permute((0,2,1))   
    
    A, lmbd_est, gamma_est, theta_est, G_est, mod_BIC, Loss, flag_maxiter = \
        ALS_spr_bound_auto_real(macro_data[:N, start_train:T_test], 
                                p, r, s, N, T_test-start_train, P, lagr_mul,max_iter,
                                lr_omega, np.array(lmbd), np.array(gamma),
                                np.array(theta), stop_thres, stop_method, A_ini = A_ini)
    L = get_L(lmbd_est, gamma_est, theta_est, N, r, s, T_test-start_train, p)
    y_predict_h[:,k:(k+1)] = tensor_op.unfold(mode_dot(G_est, L, 2), 1).numpy()@np.reshape(np.flip(macro_data[:N, start_train:T_test], axis=1), (-1, 1), order='F')
    # prid_h_err2.append(np.linalg.norm(y_predict_h[:,k]-macro_data[:N, T_test], ord=2)**2)
    # prid_h_err1.append(np.linalg.norm(y_predict_h[:,k]-macro_data[:N, T_test], ord=1))
    err_l2 = np.linalg.norm(y_predict_h[:,k]-macro_data[:N, T_test], ord=2)
    err_l1 = np.linalg.norm(y_predict_h[:,k]-macro_data[:N, T_test], ord=1)
    flags.append(flag_maxiter)

    pd.DataFrame(lmbd_est).to_csv('lmbd.csv', mode='a', index=False,header=False)
    pd.DataFrame([err_l2, err_l1]).T.to_csv('err.csv', mode='a', index=False,header=False)
    pd.DataFrame(tensor_op.unfold(G_est, 1).numpy()).to_csv('Gmat.csv', mode='a', index=False,header=False)
    pd.DataFrame(tensor_op.unfold(A[:,:,:8], 1).numpy()).to_csv('Amat.csv', mode='a', index=False,header=False)
    
print(flags)

# pd.DataFrame(prid_h_err1).T.to_csv('err_l1norm.csv', mode='a', index=False, header=False)
# pd.DataFrame(prid_h_err2).T.to_csv('err_l2normsqr.csv', mode='a', index=False, header=False)

# print(prid_h_err2)
# print(prid_h_err1) 

# pd.DataFrame(prid_h_err1).to_csv('test1.csv', mode='a', index=False, header=False)
# pd.DataFrame(prid_h_err2).to_csv('test2.csv', mode='a', index=False, header=False)

#%%
print(sum(G_est.reshape((-1))!=0))
print(sum(G_est[:,:,1].reshape((-1))!=0))

for i in range(G_est.shape[2]):
   print(np.linalg.norm(G_est[:,:,i], 'fro'))

#%%
plot_index_order = np.arange(0, 20, 1)
layer=0
labels = ['FM1','FM2','FMRNBA',\
          'FMRRA','FYFF','FYGT10',\
          'CPIAUCSL','GDP273','PSCCOMR','PWFSA',\
          'CES002','LHUR','CES275R',\
          'GDP251','GDP252','IPS10','UTL11','HSFR',\
          'EXRUS','FSPIN']
G_est_index = G_est[:,plot_index_order,layer]
G_est_index = G_est_index[plot_index_order,:]
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
heatmapp0 = sns.heatmap(G_est_index, center=0, cmap='vlag',linewidths=.5,linecolor='black',
                       xticklabels=labels, yticklabels=labels, vmin = -0.4,vmax=0.4)
heatmapp0.get_figure().savefig('macro20_Gheatmap_'+str(layer)+'.png', dpi=400, bbox_inches='tight')



#%%
plot_index_order = np.arange(0, 20, 1)
layer=3
labels = ['FM1','FM2','FMRNBA',\
          'FMRRA','FYFF','FYGT10',\
          'CPIAUCSL','GDP273','PSCCOMR','PWFSA',\
          'CES002','LHUR','CES275R',\
          'GDP251','GDP252','IPS10','UTL11','HSFR',\
          'EXRUS','FSPIN']
A_est_index = A[:,plot_index_order,layer]
A_est_index = A_est_index[plot_index_order,:]
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
heatmapp0 = sns.heatmap(A_est_index, center=0, cmap='vlag',linewidths=.5,linecolor='black',
                       xticklabels=labels, yticklabels=labels, vmin = -0.4,vmax=0.4)
heatmapp0.get_figure().savefig('macro20_Aheatmap_'+str(layer)+'.png', dpi=400, bbox_inches='tight')
