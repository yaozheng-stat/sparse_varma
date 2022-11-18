"""
Alternating least squares for omega and S, U1, U2
"""

import numpy as np
from numpy.random.mtrand import gamma
import scipy
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker as kron
from tensorly.tenalg import mode_dot
from help_func import *
from itertools import product
from scipy.special import logit, expit
import scipy.optimize as spo 

# lr_omega is the step length

def ALS(y, p, r, s, r1, r2, N, T, P, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, true_A, true_G, stop_thres, flag_true_G, stop_method):
    # Initialization
    d = p + r + 2 * s
    Loss = np.inf
    S, U1, U2 = spectral_init_A_exp(y, N, T, P, r1, r2)
    # A = init_A(y,N,T,P,r1,r2)
    lmbd = np.copy(lmbd_true[:])
    gamma = np.copy(gamma_true[:])
    theta = np.copy(theta_true[:])
    # truncated L
    L = get_L(lmbd, gamma, theta, N, r, s, P, p)
    S = get_G(S, L)
    G = mode_dot(mode_dot(S, U1, 0), U2, 1)
    # full L
    L = get_L(lmbd, gamma, theta, N, r, s, T, p)
    # use full L and G to restore a full A
    A = mode_dot(G, L, 2)
    if flag_true_G:
        S, U1, U2 = get_S_and_U(true_G[:, :, :d], r1, r2, d)
        G = mode_dot(S, np.array(U1), 0)
        G = mode_dot(G, np.array(U2), 1)

    # useful for ls
    Y = y[:, 1:]
    # Y_col = np.reshape(np.flip(Y,axis=1),(-1,1),order='F') # vectorized Y
    # Y_col = np.reshape(Y, (-1, 1), order='F')  # y2 to yt
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]
    X2 = np.zeros((T - 1, T - 1, N))
    if p == 2:
        for i in range(1, T - 1):
            X2[i, :(i + 1 - p), :] = np.flip(y[:, :(i + 1 - p)], axis=1).T
    else:
        for i in range(T - 1):
            X2[i, :(i + 1 - p), :] = np.flip(y[:, :(i + 1 - p)], axis=1).T

    # ALS steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd)
        for k in range(r):
            # update lmbd
            grad, hess = vec_jac_hess_lmbd(lmbd[k], k, G, L, Y, X1, X2, p, T)
            # grad1,hess1 = jac_hess_lmbd(lmbd[k],k,G,L,y,p,T)
            # lmbd[k] = lmbd[k] - lr_omega * jac_lmbd(lmbd[k],k,G,L,y,p,T) / hess_lmbd(lmbd[k],k,G,L,y,p,T)
            lmbd[k] = lmbd[k] - lr_omega * grad / hess
            lmbd[k] = max(min(0.9, lmbd[k]), -0.9)  # lambda is between -.9 and .9
            power_series = np.arange(1, T - p + 1)
            L[p:, p + k] = np.power(lmbd[k], power_series)
            # print("grad: ",grad)
            # print('hess: ',hess)
            # Loss = loss(y,G,L,T)
            # print('lmbd: ',Loss)
        pre_gamma = np.copy(gamma)
        pre_theta = np.copy(theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma, grad_theta, hess = vec_jac_hess_gamma_theta([gamma[k], theta[k]], k, G, L, Y, X1, X2, p, r, T)
            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            grad = np.array([grad_gamma, grad_theta])
            hess_inv = np.linalg.inv(hess)
            temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            if temp > 0.9 or temp < 0.1:
                temp = gamma[k] - lr_omega * grad_gamma
                theta[k] = theta[k] - lr_omega * grad_theta
            else:
                theta[k] = theta[k] - lr_omega * (hess_inv @ grad)[1]
            gamma[k] = max(min(0.9, temp), 0.1)
            theta[k] = max(min(np.pi / 2, theta[k]), -np.pi / 2)
            power_series = np.arange(1, T - p + 1)
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))

        print(lmbd)
        print(gamma, theta)
        # prepare L, z, Z
        # note that there are in total T-1 z_t
        # L = get_L(lmbd,gamma,theta,N,r,s,T,p)
        # z = np.zeros((N*d,T-1))
        # for t in range(1,T):
        #     x_t = x[(T-t)*N:] 
        #     z[:,t-1:t] = get_z(x_t,N,L[:t,:])

        z = kron([L[:T - 1, :].T, np.identity(N)]) @ X1

        S1 = tensor_op.unfold(S, 1)

        # update U1
        X_1 = S1 @ kron([np.identity(d), U2.T]) @ z
        U1 = (X_1 @ Y.T).T @ np.linalg.inv(X_1 @ X_1.T)
        # print("U1's norm: ",np.linalg.norm(U1,ord='fro'))

        # update U2
        X_2 = np.zeros((n * N, N * r2))
        # Z = get_Z(z,N)
        for i in range(n):
            X_2[i * N:(i + 1) * N, :] = U1 @ S1 @ kron([np.reshape(z[:, i], (N, d), order='F').T, np.identity(r2)])
        U2 = np.reshape((Y_col.T @ X_2) @ np.linalg.inv(X_2.T @ X_2), (r2, N), order='F').T
        # print("U2's norm: ",np.linalg.norm(U2,ord='fro'))

        # update S
        X_s = np.zeros((n * N, r1 * r2 * d))
        for i in range(n):
            X_s[i * N:(i + 1) * N, :] = kron([z[:, i].T @ kron([np.identity(d), U2]), U1])
        S = np.reshape(np.linalg.inv(X_s.T @ X_s) @ (X_s.T @ Y_col), (r1, r2, d), order='F')
        # print("S's norm: ",np.linalg.norm(tensor_op.unfold(S,1),ord='fro'))

        # restore G
        pre_G = G
        G = mode_dot(S, np.array(U1), 0)
        G = tl.tenalg.mode_dot(G, np.array(U2), 1)
        # print("G's norm: ",np.linalg.norm(tensor_op.unfold(G,1),ord='fro'))

        # HOSVD to make U1 U2 orthonormal
        S, U1, U2 = get_S_and_U(G, r1, r2, d)

        # early stop
        pre_A = A
        A = tl.tenalg.mode_dot(G, L, 2)
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres):
            Loss = loss(y, G, L, T)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            Loss = loss(y, G, L, T)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter
            # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
            # print("No. of iter: ", iter_no)
            # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
            # print("Warning loss: ", Loss)
            # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
            # return A,lmbd,gamma,theta,G,Loss,flag_maxiter 
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

    # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    Loss = loss(y, G, L, T)
    A = tl.tenalg.mode_dot(G, L, 2)
    return A, lmbd, gamma, theta, G, Loss, flag_maxiter


def ALS_spr(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, true_A, true_G, stop_thres, flag_true_G, stop_method):
    # y the data frame
    # p AR(p) r real eigenvealue, s complex eigenvalue pairs
    # N nrow in y; P truncated time lag; T total time tag
    # n_iter lr_omega about iteration and step length
    # lmbd_true gamma_true theta_true true_A true value of the model
    d = p + r + 2 * s
    Loss = np.inf

    A = init_A_spr(y, N, T, P)  # initialize A with truncated P
    lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    gamma = np.copy(gamma_true[:])
    theta = np.copy(theta_true[:])

    L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    G = get_G(A, L)  # initialize G

    L = get_L(lmbd, gamma, theta, N, r, s, T, p)  # get full L
    A = mode_dot(G, L, 2)  # use full L and G to restore a full A

    Y = y[:, 1:]
    Y_col = np.reshape(Y, (-1, 1), order='F')  # change Y to a vector
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))  # X1 for vectorized predictor, filled with 0
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]

    # ALS_spr steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd)
        for k in range(r):
            # update lmbd
            grad, hess = vec_jac_hess_lmbd(N, lmbd[k], k, G, L, y, Y, X1, p, T)  # get grad and hess
            # lmbd[k] = lmbd[k] - lr_omega * grad / hess  # update lambda by lr_omega step length
            lmbd[k] = lmbd[k] - lr_omega * grad
            # lmbd[k] = lmbd[k] - grad / hess
            lmbd[k] = max(min(0.9, lmbd[k]), -0.9)  # gerentee the range
            power_series = np.arange(1, T - p + 1)  # update L
            L[p:, p + k] = np.power(lmbd[k], power_series)  # update L

        pre_gamma = np.copy(gamma)
        pre_theta = np.copy(theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma, grad_theta, hess = vec_jac_hess_gamma_theta(N, [gamma[k], theta[k]], k, G, L, y, Y, X1, p, r, T)
            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            # grad = np.array([grad_gamma, grad_theta])
            # hess_inv = np.linalg.inv(hess)  
            # hess_inv = np.linalg.inv(hess)
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0] # origin
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            temp = gamma[k] - lr_omega * grad_gamma
            theta[k] = theta[k] - lr_omega * grad_theta
            '''
            if temp > 0.9 or temp < 0.1:
                temp = gamma[k] - lr_omega * grad_gamma
                theta[k] = theta[k] - lr_omega * grad_theta
            else:
                theta[k] = theta[k] - lr_omega * (hess_inv @ grad)[1]'''
            gamma[k] = max(min(0.9, temp), 0.1)
            theta[k] = max(min(np.pi / 2, theta[k]), -np.pi / 2)
            power_series = np.arange(1, T - p + 1)
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))


        # update G with soft-threshold
        
        pre_G = np.copy(G)
        for i, j, k in product(range(N), range(N), range(d)):
            grad = lasso_grd(N, i, j, k, G, L, Y, X1, T)
            G_ijk_temp = G[i, j, k] - lr_omega * grad
            G[i, j, k] = soft_threshold(lr_omega*lagr_mul, G_ijk_temp)

        # early stop
        # print(np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'))
        # print(G[0,0,:])
        
        pre_A = np.copy(A)
        A = tl.tenalg.mode_dot(G, L, 2)
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres):
            Loss = loss(y, G, L, T)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            Loss = loss(y, G, L, T)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
        # print("No. of iter: ", iter_no)
        # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("Warning loss: ", Loss)
        # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

        # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    Loss = loss(y, G, L, T)
    A = tl.tenalg.mode_dot(G, L, 2)
    return A, lmbd, gamma, theta, G, Loss, flag_maxiter

def ALS_spr_trans(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, true_A, true_G, stop_thres, flag_true_G, stop_method,ad=1):
    # y the data frame
    # p AR(p) r real eigenvealue, s complex eigenvalue pairs
    # N nrow in y; P truncated time lag; T total time tag
    # n_iter lr_omega about iteration and step length
    # lmbd_true gamma_true theta_true true_A true value of the model
    d = p + r + 2 * s
    Loss = np.inf

    A = init_A_spr(y, N, T, P)  # initialize A with truncated P
    lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    tan_lmbd = np.tan(np.multiply(lmbd, np.pi/2))
    gamma = np.copy(gamma_true[:])
    logit_gamma = np.log(gamma) - np.log([1-x for x in gamma])
    theta = np.copy(theta_true[:])
    tan_theta = np.tan(theta)
    
    L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    G = get_G(A, L)  # initialize G

    L = get_L(lmbd, gamma, theta, N, r, s, T, p)  # get full L
    A = mode_dot(G, L, 2)  # use full L and G to restore a full A

    Y = y[:, 1:]
    Y_col = np.reshape(Y, (-1, 1), order='F')  # change Y to a vector
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))  # X1 for vectorized predictor, filled with 0
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]

    # ALS_spr steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_tan_lmbd = np.copy(tan_lmbd)
        for k in range(r):
            # update lmbd
            grad, hess = vec_jac_hess_lmbd(N, lmbd[k], k, G, L, y, Y, X1, p, T)  # get grad and hess
            grad = grad/(1+tan_lmbd[k]**2)*2/np.pi
            # lmbd[k] = lmbd[k] - lr_omega * grad / hess  # update lambda by lr_omega step length
            temp_tan_lmbd = tan_lmbd[k] - lr_omega * grad/ad
            # print('tan_lambda', grad)
            tan_lmbd[k] = temp_tan_lmbd
            lmbd[k] = np.arctan(tan_lmbd[k])*2/np.pi 
           
            # lmbd[k] = lmbd[k] - grad / hess
            # lmbd[k] = max(min(0.9, temp_l), -0.9)  # gerentee the range
            power_series = np.arange(1, T - p + 1)  # update L
            L[p:, p + k] = np.power(lmbd[k], power_series)  # update L

        pre_logit_gamma = np.copy(logit_gamma)
        pre_tan_theta = np.copy(tan_theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma, grad_theta, hess = vec_jac_hess_gamma_theta(N, [gamma[k], theta[k]], k, G, L, y, Y, X1, p, r, T)
            grad_gamma = grad_gamma*np.exp(-logit_gamma[k])/(1+np.exp(-logit_gamma[k]))**2
            grad_theta = grad_theta/(1+tan_theta[k]**2)
            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            # grad = np.array([grad_gamma, grad_theta])
            # hess_inv = np.linalg.inv(hess)  
            # hess_inv = np.linalg.inv(hess)
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0] # origin
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            temp_logit_gamma = logit_gamma[k] - lr_omega * grad_gamma/ad
            temp_tan_theta = tan_theta[k] - lr_omega * grad_theta/ad

            logit_gamma[k] = temp_logit_gamma
            tan_theta[k] = temp_tan_theta
            
            gamma[k] = 1/(1+np.exp(-logit_gamma[k]))
            theta[k] = np.arctan(tan_theta[k])
               
            # gamma[k] = max(min(0.9, temp_g), 0.1)
            # theta[k] = max(min(np.pi / 2, temp_th), -np.pi / 2)
            power_series = np.arange(1, T - p + 1)
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))
            # print((lmbd, gamma, theta))

        # print(lmbd, gamma, theta)


        # update G with soft-threshold
        
        pre_G = np.copy(G)
        for i, j, k in product(range(N), range(N), range(d)):
            grad = lasso_grd(N, i, j, k, G, L, Y, X1, T)
            G_ijk_temp = G[i, j, k] - lr_omega * grad
            G[i, j, k] = soft_threshold(lr_omega*lagr_mul, G_ijk_temp)

        # early stop
        # print(np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'))
        # print(G[0,0,:])
        
        pre_A = np.copy(A)
        A = tl.tenalg.mode_dot(G, L, 2)
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([tan_lmbd - pre_tan_lmbd, logit_gamma - pre_logit_gamma, tan_theta - pre_tan_theta]), ord=2) < stop_thres):
            Loss = loss(y, G, L, T)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            Loss = loss(y, G, L, T)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
        # print("No. of iter: ", iter_no)
        # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("Warning loss: ", Loss)
        # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

        # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    Loss = loss(y, G, L, T)
    A = tl.tenalg.mode_dot(G, L, 2)
    return A, lmbd, gamma, theta, G, Loss, flag_maxiter

def ALS_spr_bound(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, true_A, true_G, stop_thres, flag_true_G, stop_method,ad=1, epsilon=0.05):
    # y the data frame
    # p AR(p) r real eigenvealue, s complex eigenvalue pairs
    # N nrow in y; P truncated time lag; T total time tag
    # n_iter lr_omega about iteration and step length
    # lmbd_true gamma_true theta_true true_A true value of the model
    d = p + r + 2 * s
    Loss = np.inf

    # A = init_A_spr(y, N, T, P)  # initialize A with truncated P
    lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    gamma = np.copy(gamma_true[:])
    theta = np.copy(theta_true[:])
    
    #L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    #G = get_G(A, L)  # initialize G
    G=np.copy(true_G)

    L = get_L(lmbd, gamma, theta, N, r, s, T, p)  # get full L
    A = mode_dot(G, L, 2)  # use full L and G to restore a full A

    Y = y[:, 1:]
    Y_col = np.reshape(Y, (-1, 1), order='F')  # change Y to a vector
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))  # X1 for vectorized predictor, filled with 0
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]

    # ALS_spr steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd)
        for k in range(r):
            # update lmbd
            grad, hess = vec_jac_hess_lmbd(N, lmbd[k], k, G, L, y, Y, X1, p, T)  # get grad and hess
            # lmbd[k] = lmbd[k] - lr_omega * grad / hess  # update lambda by lr_omega step length
            temp_lmbd = lmbd[k] - lr_omega * grad/ad
            # print('tan_lambda', grad)
            lmbd[k] = sorted([1-epsilon, -1+epsilon, temp_lmbd])[1]
           
            # lmbd[k] = lmbd[k] - grad / hess
            # lmbd[k] = max(min(0.9, temp_l), -0.9)  # gerentee the range
            power_series = np.arange(1, T - p + 1)  # update L
            L[p:, p + k] = np.power(lmbd[k], power_series)  # update L

        pre_gamma = np.copy(gamma)
        pre_theta = np.copy(theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma, grad_theta, hess = vec_jac_hess_gamma_theta(N, [gamma[k], theta[k]], k, G, L, y, Y, X1, p, r, T)

            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            # grad = np.array([grad_gamma, grad_theta])
            # hess_inv = np.linalg.inv(hess)  
            # hess_inv = np.linalg.inv(hess)
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0] # origin
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            temp_gamma = gamma[k] - lr_omega * grad_gamma/ad
            temp_theta = theta[k] - lr_omega * grad_theta/ad
            
            gamma[k] = sorted([1-epsilon, 0, temp_gamma])[1]
            theta[k] = sorted([np.pi/2-epsilon, -np.pi/2+epsilon, temp_theta])[1]
                
            

               
            # gamma[k] = max(min(0.9, temp_g), 0.1)
            # theta[k] = max(min(np.pi / 2, temp_th), -np.pi / 2)
            power_series = np.arange(1, T - p + 1)
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))
            # print((lmbd, gamma, theta))

        # print(lmbd, gamma, theta)


        # update G with soft-threshold
        
        pre_G = np.copy(G)
        for i, j, k in product(range(N), range(N), range(d)):
            grad = lasso_grd(N, i, j, k, G, L, Y, X1, T)
            G_ijk_temp = G[i, j, k] - lr_omega * grad
            G[i, j, k] = soft_threshold(lr_omega*lagr_mul, G_ijk_temp)

        # early stop
        # print(np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'))
        # print(G[0,0,:])
        
        pre_A = np.copy(A)
        A = tl.tenalg.mode_dot(G, L, 2)
        
        
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres):
            Loss = loss(y, A, T)
            BIC = T*np.log(Loss)+2*1.2*np.count_nonzero(G) # for regulation parameter
            # BIC = np.log(Loss)+1.2*(r+2*s)*np.log(d*N)*np.log(T)/T # for (p,r,s)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter, BIC
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            Loss = loss(y, A, T)
            BIC = T*np.log(Loss)+2*1.2*np.count_nonzero(G)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter, BIC
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
        # print("No. of iter: ", iter_no)
        # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("Warning loss: ", Loss)
        # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

        # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    Loss = loss(y, A, T)
    A = tl.tenalg.mode_dot(G, L, 2)
    return A, lmbd, gamma, theta, G, Loss, flag_maxiter

def ALS_spr_bound_auto(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, true_A, true_G, stop_thres, flag_true_G, stop_method, ad=1, epsilon=0.05):
    BIC_all = []
    G_all = []
    lmbd_est_all = []
    gamma_est_all = []
    theta_est_all = []
    
    for lagr_mul_single in lagr_mul:
        A, lmbd_est, gamma_est, theta_est, G_est, Loss, flag_maxiter, BIC_single = \
        ALS_spr_bound(y, p, r, s, N, T, P, lagr_mul_single, 
                           n_iter, lr_omega, lmbd_true, gamma_true, 
                           theta_true, true_A, true_G, stop_thres, flag_true_G,stop_method)
        BIC_all.append(BIC_single)
        G_all.append(G_est)
        lmbd_est_all.append(lmbd_est)
        gamma_est_all.append(gamma_est)
        theta_est_all.append(theta_est)
    
    best_BIC = np.argmin(BIC_all)
    lmbd = lmbd_est_all[best_BIC]
    gamma = gamma_est_all[best_BIC]
    theta = theta_est_all[best_BIC]
    G = G_all[best_BIC]
    
    return A, lmbd, gamma, theta, G, Loss, flag_maxiter

def ALS_spr_bound_exp2(y, p, r, s, N, T, P, B, B_inv, lagr_mul, n_iter, lr_omega, stop_thres, flag_true_G, stop_method,ad=1, epsilon=0.05):
    # y the data frame
    # p AR(p) r real eigenvealue, s complex eigenvalue pairs
    # N nrow in y; P truncated time lag; T total time tag
    # n_iter lr_omega about iteration and step length
    # lmbd_true gamma_true theta_true true_A true value of the model
    d = p + r + 2 * s
    Loss = np.inf
    #P = int(np.floor(1.5*T**(1/2)))
    #A = init_A_spr_var(y, p, r ,s, N, T, P, lagr_mul) # initialize A with sparse var
    lmbd = np.repeat(-0.7,r)
    gamma = np.repeat(0.7,s)
    theta = np.repeat(np.pi/3,s)
    
    Theta = get_Theta(N,lmbd,gamma,theta)
    
    if p==0:
        G = B_to_G_1(r,s,Theta, B, B_inv)
    elif p==1:
        Psi = np.diag(np.repeat(0.5,N))
        G = B_to_G_11(p,r,s,Psi,Theta, B, B_inv)
         
    #lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    #gamma = np.copy(gamma_true[:])
    #theta = np.copy(theta_true[:])
    
    #L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    #G = get_G(A, L)  # initialize G
    #G=np.copy(true_G)

    L = get_L(lmbd, gamma, theta, N, r, s, T, p)  # get full L
    A = mode_dot(G, L, 2)  # use full L and G to restore a full A

    Y = y[:, 1:]
    Y_col = np.reshape(Y, (-1, 1), order='F')  # change Y to a vector
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))  # X1 for vectorized predictor, filled with 0
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]

    # ALS_spr steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd)
        for k in range(r):
            # update lmbd
            grad, hess = vec_jac_hess_lmbd(N, lmbd[k], k, G, L, y, Y, X1, p, T)  # get grad and hess
            # lmbd[k] = lmbd[k] - lr_omega * grad / hess  # update lambda by lr_omega step length
            temp_lmbd = lmbd[k] - lr_omega * grad/ad
            # print('tan_lambda', grad)
            lmbd[k] = sorted([1-epsilon, -1+epsilon, temp_lmbd])[1]
           
            # lmbd[k] = lmbd[k] - grad / hess
            # lmbd[k] = max(min(0.9, temp_l), -0.9)  # gerentee the range
            power_series = np.arange(1, T - p + 1)  # update L
            L[p:, p + k] = np.power(lmbd[k], power_series)  # update L

        pre_gamma = np.copy(gamma)
        pre_theta = np.copy(theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma, grad_theta, hess = vec_jac_hess_gamma_theta(N, [gamma[k], theta[k]], k, G, L, y, Y, X1, p, r, T)

            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            # grad = np.array([grad_gamma, grad_theta])
            # hess_inv = np.linalg.inv(hess)  
            # hess_inv = np.linalg.inv(hess)
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0] # origin
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            temp_gamma = gamma[k] - lr_omega * grad_gamma/ad
            temp_theta = theta[k] - lr_omega * grad_theta/ad
            
            gamma[k] = sorted([1-epsilon, epsilon, temp_gamma])[1]
            theta[k] = sorted([np.pi/2-epsilon, -np.pi/2+epsilon, temp_theta])[1]
                
            

               
            # gamma[k] = max(min(0.9, temp_g), 0.1)
            # theta[k] = max(min(np.pi / 2, temp_th), -np.pi / 2)
            power_series = np.arange(1, T - p + 1)
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))
            # print((lmbd, gamma, theta))

        print(lmbd, gamma, theta)


        # update G with soft-threshold
        
        pre_G = np.copy(G)
        for i, j, k in product(range(N), range(N), range(d)):
            grad = lasso_grd(N, i, j, k, G, L, Y, X1, T)
            G_ijk_temp = G[i, j, k] - lr_omega * grad
            G[i, j, k] = soft_threshold(lr_omega*lagr_mul, G_ijk_temp)

        # early stop
        # print(np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'))
        # print(G[0,0,:])
        
        pre_A = np.copy(A)
        A = tl.tenalg.mode_dot(G, L, 2)
        
        
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres):
            Loss = loss(y, A, T)
            BIC = T*np.log(Loss)+2*1.2*np.count_nonzero(G) # for regulation parameter
            mod_BIC = [np.log(Loss), (r+2*s)*np.log(d*N)*np.log(T)/T]
            # BIC = np.log(Loss)+1.2*(r+2*s)*np.log(d*N)*np.log(T)/T # for (p,r,s)
            print("No. of iter: ", iter_no)
            #print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter, BIC, mod_BIC
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            Loss = loss(y, A, T)
            BIC = T*np.log(Loss)+2*1.2*np.count_nonzero(G)
            mod_BIC = [np.log(Loss), (r+2*s)*np.log(d*N)*np.log(T)/T]
            print("No. of iter: ", iter_no)
            #print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter, BIC, mod_BIC
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
        # print("No. of iter: ", iter_no)
        # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("Warning loss: ", Loss)
        # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

        # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    Loss = loss(y, A, T)
    A = tl.tenalg.mode_dot(G, L, 2)
    BIC = T*np.log(Loss)+2*1.2*np.count_nonzero(G)
    mod_BIC = [np.log(Loss), (r+2*s)*np.log(d*N)*np.log(T)/T]
    return A, lmbd, gamma, theta, G, Loss, flag_maxiter, BIC, mod_BIC

def ALS_spr_bound_auto_exp2(y, p, r, s, N, T, P, B, B_inv, lagr_mul, n_iter, lr_omega, stop_thres, flag_true_G, stop_method, ad=1, epsilon=0.05):
    BIC_all = []
    G_all = []
    lmbd_est_all = []
    gamma_est_all = []
    theta_est_all = []
    mod_BIC_all = []
    
    for lagr_mul_single in lagr_mul:
        A, lmbd_est, gamma_est, theta_est, G_est, Loss, flag_maxiter, BIC_single, mod_BIC_single = \
        ALS_spr_bound_exp2(y, p, r, s, N, T, P, B, B_inv, lagr_mul_single,
                           n_iter, lr_omega, stop_thres, flag_true_G,stop_method)
        BIC_all.append(BIC_single)
        G_all.append(G_est)
        lmbd_est_all.append(lmbd_est)
        gamma_est_all.append(gamma_est)
        theta_est_all.append(theta_est)
        mod_BIC_all.append(mod_BIC_single)
    
    best_BIC = np.argmin(BIC_all)
    lmbd = lmbd_est_all[best_BIC]
    gamma = gamma_est_all[best_BIC]
    theta = theta_est_all[best_BIC]
    G = G_all[best_BIC]
    mod_BIC = mod_BIC_all[best_BIC]
    
    return A, lmbd, gamma, theta, G, mod_BIC, Loss, flag_maxiter


def ALS_spr_real(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, stop_thres, stop_method,ad=1):
    # y the data frame
    # p AR(p) r real eigenvealue, s complex eigenvalue pairs
    # N nrow in y; P truncated time lag; T total time tag
    # n_iter lr_omega about iteration and step length
    # lmbd_true gamma_true theta_true true_A true value of the model
    d = p + r + 2 * s

    A = init_A_spr(y, N, T, P)  # initialize A with truncated P
    lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    gamma = np.copy(gamma_true[:])
    theta = np.copy(theta_true[:])

    L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    G = get_G(A, L)  # initialize G

    L = get_L(lmbd, gamma, theta, N, r, s, T, p)  # get full L
    A = mode_dot(G, L, 2)  # use full L and G to restore a full A

    Y = y[:, 1:]
    Y_col = np.reshape(Y, (-1, 1), order='F')  # change Y to a vector
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))  # X1 for vectorized predictor, filled with 0
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]

    # ALS_spr steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd)
        for k in range(r):
            # update lmbd
            grad, hess = vec_jac_hess_lmbd(N, lmbd[k], k, G, L, y, Y, X1, p, T)  # get grad and hess
            # lmbd[k] = lmbd[k] - lr_omega * grad / hess  # update lambda by lr_omega step length
            temp_l = lmbd[k] - lr_omega * grad/ad
            print('lambda', grad)
            if temp_l > 0.99 or temp_l < -0.99:
                lmbd[k] = 0.5
            else:
                lmbd[k] = temp_l
           
            # lmbd[k] = lmbd[k] - grad / hess
            # lmbd[k] = max(min(0.9, temp_l), -0.9)  # gerentee the range
            power_series = np.arange(1, T - p + 1)  # update L
            L[p:, p + k] = np.power(lmbd[k], power_series)  # update L

        pre_gamma = np.copy(gamma)
        pre_theta = np.copy(theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma, grad_theta, hess = vec_jac_hess_gamma_theta(N, [gamma[k], theta[k]], k, G, L, y, Y, X1, p, r, T)
            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            # grad = np.array([grad_gamma, grad_theta])
            # hess_inv = np.linalg.inv(hess)  
            # hess_inv = np.linalg.inv(hess)
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0] # origin
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            temp_g = gamma[k] - lr_omega * grad_gamma/ad
            temp_th = theta[k] - lr_omega * grad_theta/ad
            print('gamma', grad_gamma)
            '''
            if temp > 0.9 or temp < 0.1:
                temp = gamma[k] - lr_omega * grad_gamma
                theta[k] = theta[k] - lr_omega * grad_theta
            else:
                theta[k] = theta[k] - lr_omega * (hess_inv @ grad)[1]'''
            
            if temp_g > 0.99 or temp_g < 0.01:
                gamma[k] = 0.5
            else:
                gamma[k] = temp_g
            
            if temp_th > np.pi / 2 or temp_th < -np.pi / 2:
                theta[k] = np.pi/3
            else:
                theta[k] = temp_th
               
            # gamma[k] = max(min(0.9, temp_g), 0.1)
            # theta[k] = max(min(np.pi / 2, temp_th), -np.pi / 2)
            power_series = np.arange(1, T - p + 1)
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))
            # print((lmbd, gamma, theta))

        print(lmbd, gamma, theta)
        # update G with soft-threshold
        
        pre_G = np.copy(G)
        for i, j, k in product(range(N), range(N), range(d)):
            grad = lasso_grd(N, i, j, k, G, L, Y, X1, T)
            # print('G', grad)
            G_ijk_temp = G[i, j, k] - lr_omega * grad
            G[i, j, k] = soft_threshold(lr_omega*lagr_mul, G_ijk_temp)
            # print('G',grad)

        # early stop
        # print(np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'))
        # print(G[0,0,:])
        
        pre_A = np.copy(A)
        A = tl.tenalg.mode_dot(G, L, 2)
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres):
            print("No. of iter: ", iter_no)   
            return A, lmbd, gamma, theta, G, flag_maxiter
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):

            print("No. of iter: ", iter_no)

            return A, lmbd, gamma, theta, G, flag_maxiter
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
        # print("No. of iter: ", iter_no)
        # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("Warning loss: ", Loss)
        # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

        # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    A = tl.tenalg.mode_dot(G, L, 2)
    return A, lmbd, gamma, theta, G, flag_maxiter

def ALS_spr_trans_real(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, stop_thres, stop_method, ad=1):
    # y the data frame
    # p AR(p) r real eigenvealue, s complex eigenvalue pairs
    # N nrow in y; P truncated time lag; T total time tag
    # n_iter lr_omega about iteration and step length
    # lmbd_true gamma_true theta_true true_A true value of the model
    d = p + r + 2 * s

    A = init_A_spr(y, N, T, P)  # initialize A with truncated P
    lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    tan_lmbd = np.tan(np.multiply(lmbd, np.pi/2))
    gamma = np.copy(gamma_true[:])
    logit_gamma = np.log(gamma) - np.log([1-x for x in gamma])
    theta = np.copy(theta_true[:])
    tan_theta = np.tan(theta)
    
    L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    G = get_G(A, L)  # initialize G

    L = get_L(lmbd, gamma, theta, N, r, s, T, p)  # get full L
    A = mode_dot(G, L, 2)  # use full L and G to restore a full A

    Y = y[:, 1:]
    Y_col = np.reshape(Y, (-1, 1), order='F')  # change Y to a vector
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))  # X1 for vectorized predictor, filled with 0
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]

    # ALS_spr steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_tan_lmbd = np.copy(tan_lmbd)
        for k in range(r):
            # update lmbd
            grad, hess = vec_jac_hess_lmbd(N, lmbd[k], k, G, L, y, Y, X1, p, T)  # get grad and hess
            grad = grad/(1+tan_lmbd[k]**2)*2/np.pi
            # lmbd[k] = lmbd[k] - lr_omega * grad / hess  # update lambda by lr_omega step length
            temp_tan_lmbd = tan_lmbd[k] - lr_omega * grad/ad
            print('tan_lambda', grad)
            tan_lmbd[k] = temp_tan_lmbd
            lmbd[k] = np.arctan(tan_lmbd[k])*2/np.pi 
           
            # lmbd[k] = lmbd[k] - grad / hess
            # lmbd[k] = max(min(0.9, temp_l), -0.9)  # gerentee the range
            power_series = np.arange(1, T - p + 1)  # update L
            L[p:, p + k] = np.power(lmbd[k], power_series)  # update L

        pre_logit_gamma = np.copy(logit_gamma)
        pre_tan_theta = np.copy(tan_theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma, grad_theta, hess = vec_jac_hess_gamma_theta(N, [gamma[k], theta[k]], k, G, L, y, Y, X1, p, r, T)
            grad_gamma = grad_gamma*np.exp(-logit_gamma[k])/(1+np.exp(-logit_gamma[k]))**2
            grad_theta = grad_theta/(1+tan_theta[k]**2)
            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            # grad = np.array([grad_gamma, grad_theta])
            # hess_inv = np.linalg.inv(hess)  
            # hess_inv = np.linalg.inv(hess)
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0] # origin
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            temp_logit_gamma = logit_gamma[k] - lr_omega * grad_gamma/ad
            temp_tan_theta = tan_theta[k] - lr_omega * grad_theta/ad

            logit_gamma[k] = temp_logit_gamma
            tan_theta[k] = temp_tan_theta
            
            gamma[k] = 1/(1+np.exp(-logit_gamma[k]))
            theta[k] = np.arctan(tan_theta[k])
               
            # gamma[k] = max(min(0.9, temp_g), 0.1)
            # theta[k] = max(min(np.pi / 2, temp_th), -np.pi / 2)
            power_series = np.arange(1, T - p + 1)
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))
            # print((lmbd, gamma, theta))

        print(lmbd, gamma, theta)
        # update G with soft-threshold
        
        def soft_threshold_all(beta):
            if beta > lr_omega*lagr_mul:
                return beta - lr_omega*lagr_mul
            elif beta < -lr_omega*lagr_mul:
                return beta + lr_omega*lagr_mul
            else:
                return 0.0
        
        pre_G = np.copy(G)
        #print(pre_G)
        z = kron([L[:T-1,:].T,np.identity(N)]) @ X1
        X_g = np.zeros((n*N,N*N*d))
        for i in range(n):
            X_g[i*N:(i+1)*N,:] = kron([z[:,i].T @ kron([np.identity(d), np.identity(N)]), np.identity(N)])
           
        G_vec = tensor_op.unfold(G, 1).numpy().reshape((-1,1), order = 'F')
        G_grad_vec = 2*X_g.T@(X_g@G_vec-Y_col)/T
        
        G_vec_update = (G_vec - lr_omega*G_grad_vec).reshape((-1))
        
        #print(G_vec_update)
        
        G = np.array(list(map(soft_threshold_all, G_vec_update))).reshape((N,N,d), order='F')

        # early stop
        # print(np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'))
        # print(G[0,0,:])
        
        pre_A = np.copy(A)
        A = tl.tenalg.mode_dot(G, L, 2)
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([tan_lmbd - pre_tan_lmbd, logit_gamma - pre_logit_gamma, tan_theta - pre_tan_theta]), ord=2) < stop_thres):
            print("No. of iter: ", iter_no)   
            return A, lmbd, gamma, theta, G, flag_maxiter
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):

            print("No. of iter: ", iter_no)

            return A, lmbd, gamma, theta, G, flag_maxiter
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
        # print("No. of iter: ", iter_no)
        # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("Warning loss: ", Loss)
        # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

        # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    A = tl.tenalg.mode_dot(G, L, 2)
    return A, lmbd, gamma, theta, G, flag_maxiter

def ALS_spr_defult_real(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, stop_thres, stop_method, ad=1):
    # y the data frame
    # p AR(p) r real eigenvealue, s complex eigenvalue pairs
    # N nrow in y; P truncated time lag; T total time tag
    # n_iter lr_omega about iteration and step length
    # lmbd_true gamma_true theta_true true_A true value of the model
    d = p + r + 2 * s

    A = init_A_spr(y, N, T, P)  # initialize A with truncated P
    lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    gamma = np.copy(gamma_true[:])
    theta = np.copy(theta_true[:])
    
    L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    G = get_G(A, L)  # initialize G

    L = get_L(lmbd, gamma, theta, N, r, s, T, p)  # get full L
    A = mode_dot(G, L, 2)  # use full L and G to restore a full A

    Y = y[:, 1:]
    Y_col = np.reshape(Y, (-1, 1), order='F')  # change Y to a vector
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))  # X1 for vectorized predictor, filled with 0
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]

    # ALS_spr steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd)
        pre_gamma = np.copy(gamma)
        pre_theta = np.copy(theta)
        
        lgt = np.concatenate((lmbd,gamma,theta),axis=0)
        rsg = (r,s, G, N, P, p, y, T)
        
        bnds = ((-1,1), (0,1), (-np.pi/2, np.pi/2))
        res = spo.minimize(loss_lgt, lgt, rsg, method='SLSQP', bounds=bnds)
        
        lmbd = res[0:r]
        gamma = res[r:(r+s)]
        theta = res[(r+s):(r+2*s)]
        L = get_L(lmbd, gamma, theta, N, r, s, P, p)
        
        print(lmbd, gamma, theta)
        # update G with soft-threshold
        
        pre_G = np.copy(G)
        for i, j, k in product(range(N), range(N), range(d)):
            grad = lasso_grd(N, i, j, k, pre_G, L, Y, X1, T)
            # print('G', grad)
            G_ijk_temp = G[i, j, k] - lr_omega * grad
            G[i, j, k] = soft_threshold(lr_omega*lagr_mul, G_ijk_temp)
            # print('G',grad)

        # early stop
        # print(np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'))
        # print(G[0,0,:])
        
        pre_A = np.copy(A)
        A = tl.tenalg.mode_dot(G, L, 2)
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres):
            print("No. of iter: ", iter_no)   
            return A, lmbd, gamma, theta, G, flag_maxiter
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):

            print("No. of iter: ", iter_no)

            return A, lmbd, gamma, theta, G, flag_maxiter
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
        # print("No. of iter: ", iter_no)
        # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("Warning loss: ", Loss)
        # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

        # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    A = tl.tenalg.mode_dot(G, L, 2)
    return A, lmbd, gamma, theta, G, flag_maxiter


def ALS_spr_bound_real(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega, lmbd_true, gamma_true, theta_true, \
                       stop_thres, stop_method,ad=1, epsilon=0.05,regu_c=1.2, A_ini=0):
    # y the data frame
    # p AR(p) r real eigenvealue, s complex eigenvalue pairs
    # N nrow in y; P truncated time lag; T total time tag
    # n_iter lr_omega about iteration and step length
    # lmbd_true gamma_true theta_true true_A true value of the model
    d = p + r + 2 * s
    Loss = np.inf
    #P = int(np.floor(1.5*T**(1/2)))
    if isinstance(A_ini, int):
        A = init_A_spr_var(y, p, r ,s, N, T, P, lagr_mul)  # initialize A with truncated P sparse var
    else:
        A = A_ini
    
    lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    gamma = np.copy(gamma_true[:])
    theta = np.copy(theta_true[:])
         
    #lmbd = np.copy(lmbd_true[:])  # initialize eigenvalues with true value
    #gamma = np.copy(gamma_true[:])
    #theta = np.copy(theta_true[:])
    
    #L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    #G = get_G(A, L)  # initialize G
    #G=np.copy(true_G)

    L = get_L(lmbd, gamma, theta, N, r, s, P, p)  # get truncated L to get G
    G = get_G(A, L)  # initialize G

    L = get_L(lmbd, gamma, theta, N, r, s, T, p)  # get full L
    A = mode_dot(G, L, 2)  # use full L and G to restore a full A

    Y = y[:, 1:]
    Y_col = np.reshape(Y, (-1, 1), order='F')  # change Y to a vector
    n = T - 1
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  # vectorized y, yt to y1
    X1 = np.zeros((N * (T - 1), T - 1))  # X1 for vectorized predictor, filled with 0
    for i in range(T - 1):
        X1[:(i + 1) * N, i:i + 1] = x[(T - i - 1) * N:]

    # ALS_spr steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd)
        for k in range(r):
            # update lmbd
            grad = vec_jac_hess_lmbd(N, pre_lmbd[k], k, G, L, y, Y, X1, p, T)  # get grad and hess
            # lmbd[k] = lmbd[k] - lr_omega * grad / hess  # update lambda by lr_omega step length
            temp_lmbd = lmbd[k] - lr_omega * grad/ad
            # print('tan_lambda', grad)
            lmbd[k] = sorted([1-epsilon, -1+epsilon, temp_lmbd])[1]
           
            # lmbd[k] = lmbd[k] - grad / hess
            # lmbd[k] = max(min(0.9, temp_l), -0.9)  # gerentee the range
            power_series = np.arange(1, T - p + 1)  # update L
            L[p:, p + k] = np.power(lmbd[k], power_series)  # update L

        pre_gamma = np.copy(gamma)
        pre_theta = np.copy(theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma, grad_theta = vec_jac_hess_gamma_theta(N, [pre_gamma[k], pre_theta[k]], k, G, L, y, Y, X1, p, r, T)

            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            # grad = np.array([grad_gamma, grad_theta])
            # hess_inv = np.linalg.inv(hess)  
            # hess_inv = np.linalg.inv(hess)
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0] # origin
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            temp_gamma = gamma[k] - lr_omega * grad_gamma/ad
            temp_theta = theta[k] - lr_omega * grad_theta/ad
            
            gamma[k] = sorted([1-epsilon, epsilon, temp_gamma])[1]
            theta[k] = sorted([np.pi/2-epsilon, -np.pi/2+epsilon, temp_theta])[1]
                
            

               
            # gamma[k] = max(min(0.9, temp_g), 0.1)
            # theta[k] = max(min(np.pi / 2, temp_th), -np.pi / 2)
            power_series = np.arange(1, T - p + 1)
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))
            # print((lmbd, gamma, theta))

        print(lmbd, gamma, theta)


        # update G with soft-threshold
        
        def soft_threshold_all(beta):
            if beta > lr_omega*lagr_mul:
                return beta - lr_omega*lagr_mul
            elif beta < -lr_omega*lagr_mul:
                return beta + lr_omega*lagr_mul
            else:
                return 0.0
        
        pre_G = np.copy(G)
        #print(pre_G)
        z = kron([L[:T-1,:].T,np.identity(N)]) @ X1
        X_g = np.zeros((n*N,N*N*d))
        for i in range(n):
            X_g[i*N:(i+1)*N,:] = kron([z[:,i].T @ kron([np.identity(d), np.identity(N)]), np.identity(N)])
           
        G_vec = tensor_op.unfold(G, 1).numpy().reshape((-1,1), order = 'F')
        G_grad_vec = 2*X_g.T@(X_g@G_vec-Y_col)/T
        
        G_vec_update = (G_vec - lr_omega*G_grad_vec).reshape((-1))
        
        #print(G_vec_update)
        
        G = np.array(list(map(soft_threshold_all, G_vec_update))).reshape((N,N,d), order='F')
        # early stop
        # print(np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'))
        # print(G[0,0,:])
        
        pre_A = np.copy(A)
        A = tl.tenalg.mode_dot(G, L, 2)
        
        
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres):
            Loss = loss(y, A, T)
            BIC = T*np.log(Loss)+2*regu_c*np.count_nonzero(G) # for regulation parameter
            mod_BIC = [np.log(Loss), (r+2*s)*np.log(d*N)*np.log(T)/T]
            # BIC = np.log(Loss)+1.2*(r+2*s)*np.log(d*N)*np.log(T)/T # for (p,r,s)
            print("No. of iter: ", iter_no)
            #print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter, BIC, mod_BIC
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            Loss = loss(y, A, T)
            BIC = T*np.log(Loss)+2*regu_c*np.count_nonzero(G)
            mod_BIC = [np.log(Loss), (r+2*s)*np.log(d*N)*np.log(T)/T]
            print("No. of iter: ", iter_no)
            #print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A - true_A[:, :, :T], 1), ord='fro'))
            print("Final loss: ", Loss)
            return A, lmbd, gamma, theta, G, Loss, flag_maxiter, BIC, mod_BIC
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
        # print("No. of iter: ", iter_no)
        # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("Warning loss: ", Loss)
        # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if iter_no%1 == 0:
        # print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
        # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
        # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
        # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
        # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
        # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
        # print(lmbd)
        # print("loss: ", Loss)

        # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print("No. of iter: ", iter_no)
    flag_maxiter = 1
    Loss = loss(y, A, T)
    A = tl.tenalg.mode_dot(G, L, 2)
    BIC = T*np.log(Loss)+2*regu_c*np.count_nonzero(G)
    mod_BIC = [np.log(Loss), (r+2*s)*np.log(d*N)*np.log(T)/T]
    return A, lmbd, gamma, theta, G, Loss, flag_maxiter, BIC, mod_BIC

def ALS_spr_bound_auto_real(y, p, r, s, N, T, P, lagr_mul, n_iter, lr_omega,lmbd_true, gamma_true, theta_true,\
                            stop_thres, stop_method, ad=1, epsilon=0.05, regu_c=1.2,A_ini=0):
    BIC_all = []
    G_all = []
    lmbd_est_all = []
    gamma_est_all = []
    theta_est_all = []
    mod_BIC_all = []
    
    for lagr_mul_single in lagr_mul:
        A, lmbd_est, gamma_est, theta_est, G_est, Loss, flag_maxiter, BIC_single, mod_BIC_single = \
        ALS_spr_bound_real(y, p, r, s, N, T, P, lagr_mul_single,
                           n_iter, lr_omega, lmbd_true, gamma_true, theta_true, \
                           stop_thres, stop_method, ad=ad, regu_c=regu_c, A_ini=A_ini)
        BIC_all.append(BIC_single)
        G_all.append(G_est)
        lmbd_est_all.append(lmbd_est)
        gamma_est_all.append(gamma_est)
        theta_est_all.append(theta_est)
        mod_BIC_all.append(mod_BIC_single)
    
    best_BIC = np.argmin(BIC_all)
    lmbd = lmbd_est_all[best_BIC]
    gamma = gamma_est_all[best_BIC]
    theta = theta_est_all[best_BIC]
    G = G_all[best_BIC]
    mod_BIC = mod_BIC_all[best_BIC]
    
    return A, lmbd, gamma, theta, G, mod_BIC, Loss, flag_maxiter