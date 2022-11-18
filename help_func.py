"""
Model parameters:
p: AR part order
q: MA part order
r: no. of real eigenvalues
s: no. of complex eigenvalue pairs
d = r+2s
r1: rank of first mode of A
r2: rank of second mode of A
N: dimension of the time series vector
T: length of the time series

Variables:
A: N*N*inf tensor
G: N*N*d tensor, G = S*U1*U2
S: r1*r2*d tensor
U1: N*r1 matrix
U2: N*r2 matrix
L: inf*d matrix
w: d*1 vector
lmbd: r*1 vector
gamma: s*1 vector
theta: s*1 vector

Data:
y: N*T matrix, stored as column vectors, from old (left) to new (right)
"""

# from math import sqrt
import numpy as np
from numpy.random.mtrand import gamma
import scipy
from tensorOp import tensor_op
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker as kron
from tensorly.tenalg import mode_dot
import matplotlib.pyplot as plt
from sklearn import linear_model
import torch


##################
# Initialization #
##################

def init_A(y, N, T, P, r1, r2):
    """
    Use OLS method to initialize the tensor A
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    # create X (regressors)
    X = np.zeros((N * P, T - P))
    for i in range(P):
        X[i * N:i * N + N, :] = y[:, P - i - 1:T - i - 1]
    # create Y (response)
    Y = y[:, P:]
    # solve OLS, the order of product is diff with olm
    A = (X @ Y.T).T @ np.linalg.inv(X @ X.T)
    # fold A into a tensor
    A = np.array(tensor_op.fold(A, (N, N, P), 1))
    # HOOI to get a low rank version
    A, U = tucker(A, rank=[r1, r2, P])
    A = tl.tenalg.multi_mode_dot(A, U)
    return A


def init_A_spr(y, N, T, P):
    """
    Use OLS method to initialize the tensor A
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    # create X (regressors)
    X = np.zeros((N * P, T - P))
    for i in range(P):
        X[i * N:i * N + N, :] = y[:, P - i - 1:T - i - 1]
    # create Y (response)
    Y = y[:, P:]
    # solve OLS, the order of product is diff with olm
    A = (X @ Y.T).T @ np.linalg.inv(X @ X.T)
    # fold A into a tensor
    A = np.array(tensor_op.fold(A, (N, N, P), 1))
    return A

def init_A_spr_var(y, p,r,s,N, T, P, lagr_mul):
    vec_y = np.ravel(y[:,P:].reshape((-1,1)))
    linear_matrix_1 = np.zeros((T-P,N*P))
    for i in range(T-P):
        linear_matrix_1[i,:] = y[:,i:(i+P)].T.reshape((1,-1))
    linear_matrix = np.kron(np.eye(N), linear_matrix_1)
    
    lasso = linear_model.Lasso(alpha=lagr_mul/P/(p+r+2*s), max_iter=10000)
    lasso.fit(linear_matrix, vec_y)
    lasso_coef = lasso.coef_
                     
    A = torch.from_numpy(lasso_coef.reshape((N,P,N))).permute((0,2,1))
                     
    return A

def get_Theta(N,lmbd,gamma,theta):
    stack = 0
    Theta = np.zeros((N,N))
    for r_stack in lmbd:
        Theta[stack,stack] = r_stack
        stack+=1
    
    for s_stack in range(len(gamma)):
        Theta[stack,stack] = gamma[s_stack]*np.cos(theta[s_stack])
        Theta[stack,stack+1] = gamma[s_stack]*np.sin(theta[s_stack])
        Theta[stack+1,stack] = -gamma[s_stack]*np.sin(theta[s_stack])
        Theta[stack+1,stack+1] = gamma[s_stack]*np.cos(theta[s_stack])
        stack+=2
        
    return Theta

def B_to_G_1(r,s,Theta, B, B_inv):
    d = r+2*s
    N = Theta.shape[0]
    B_ = -B_inv
    G = np.zeros((N,N,d))
    
    for r_stack in range(r):
        G[:,:,r_stack] = B[:,r_stack:r_stack+1] @ B_[r_stack:r_stack+1,:]
        
    for s_stack in range(s):
        G[:,:,r+2*s_stack] = \
            B[:,(r+2*s_stack):(r+2*s_stack+1)] @ B_[(r+2*s_stack):(r+2*s_stack+1),:] + \
            B[:,(r+2*s_stack+1):(r+2*s_stack+2)] @ B_[(r+2*s_stack+1):(r+2*s_stack+2),:]
        G[:,:,r+2*s_stack+1] = \
            B[:,(r+2*s_stack):(r+2*s_stack+1)] @ B_[(r+2*s_stack+1):(r+2*s_stack+2),:] - \
            B[:,(r+2*s_stack+1):(r+2*s_stack+2)] @ B_[(r+2*s_stack):(r+2*s_stack+1),:]
        
    return G

def B_to_G_11(p,r,s,Psi,Theta, B, B_inv):
    d = p+r+2*s
    N=Psi.shape[0]
    B_ = B_inv@(Psi-Theta)
    G = np.zeros((N,N,d))
    G[:,:,0]=Psi-Theta
    
    for r_stack in range(r):
        G[:,:,1+r_stack] = B[:,r_stack:r_stack+1] @ B_[r_stack:r_stack+1,:]
        
    for s_stack in range(s):
        G[:,:,1+r+2*s_stack] = \
            B[:,(r+2*s_stack):(r+2*s_stack+1)] @ B_[(r+2*s_stack):(r+2*s_stack+1),:] + \
            B[:,(r+2*s_stack+1):(r+2*s_stack+2)] @ B_[(r+2*s_stack+1):(r+2*s_stack+2),:]
        G[:,:,1+r+2*s_stack+1] = \
            B[:,(r+2*s_stack):(r+2*s_stack+1)] @ B_[(r+2*s_stack+1):(r+2*s_stack+2),:] - \
            B[:,(r+2*s_stack+1):(r+2*s_stack+2)] @ B_[(r+2*s_stack):(r+2*s_stack+1),:]
        
    return G
    
def spectral_init_A(y, N, T, P, r1, r2):
    """
    Spectral initialization
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    # create X (regressors)
    X = np.zeros((N * P, T - P))
    for i in range(P):
        X[i * N:i * N + N, :] = y[:, P - i - 1:T - i - 1]
    # create Y (response)
    Y = y[:, P:]
    # spectral initialization
    A = np.zeros((N, N * P))
    for t in range(T - P):
        A = A + np.outer(Y[:, t], X[:, t])
    A = A / (T - P)
    # fold A into a tensor
    A = np.array(tensor_op.fold(A, (N, N, P), 1))

    # HOOI to get a low rank version
    A, U = tucker(A, rank=[r1, r2, P])
    A = tl.tenalg.multi_mode_dot(A, U)
    return A


def spectral_init_A_exp(y, N, T, P, r1, r2, iter):
    """
    Spectral initialization
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    # create X (regressors)
    X = np.zeros((N * P, T - P))
    for i in range(P):
        X[i * N:i * N + N, :] = y[:, P - i - 1:T - i - 1]
    # create Y (response)
    Y = y[:, P:]
    # spectral initialization
    A = np.zeros((N, N * P))
    for t in range(T - P):
        A = A + np.outer(Y[:, t], X[:, t])
    A = A / (T - P)
    # fold A into a tensor
    A = np.array(tensor_op.fold(A, (N, N, P), 1))
    """
    for experiment
    """
    A1 = tensor_op.unfold(A, 1)
    _, d, _ = np.linalg.svd(A1)
    print("A1: ", d)
    plt.figure()
    plt.plot(d)
    # plt.savefig('fig/'+str(iter)+'A1')

    A2 = tensor_op.unfold(A, 2)
    _, d, _ = np.linalg.svd(A2)
    print("A2: ", d)
    plt.plot(d)
    plt.savefig('fig/' + str(iter))

    # HOOI to get a low rank version
    A, U = tucker(A, rank=[r1, r2, P])
    A = tl.tenalg.multi_mode_dot(A, U)
    return A


# generate eigenvalues, lmbd for real; gamma theta for complex
def rand_w(r, s):  # checked
    """
    Uniform distribution for now
    (may need to adjust range for endpoint issue)
    """
    lmbd = np.random.rand(r) * 2 - 1  # [-1,1]
    gamma = np.random.rand(s)  # [0,1]
    theta = np.random.rand(s) * np.pi - np.pi / 2  # [-pi/2,pi/2]
    return (lmbd, gamma, theta)


def get_L_MA(lmbd, gamma, theta, N, r, s, P):  # checked
    """
    Compute the L_MA matrix given the parameters
    Set size to be P (truncated)
    """
    L = np.zeros((P, r + 2 * s))
    tri_series = np.zeros((P, 2 * s))
    for i in range(P):
        tri_series[i, ::2] = np.cos((i + 1) * np.array(theta))
        tri_series[i, 1::2] = np.sin((i + 1) * np.array(theta))
        for j in range(r):
            L[i, j] = np.power(lmbd[j], i + 1)
        for j in range(s):
            L[i, r + 2 * j:r + 2 * j + 2] = np.power(gamma[j], i + 1)
    # np.concatenate return a combined matrix
    # np.einsum('ij,ij->ij', A, B) reutrn the dot prod of A and B
    new = np.concatenate([L[:, :r], np.einsum('ij,ij -> ij', L[:, r:], tri_series)], axis=1)
    return new


def get_L(lmbd, gamma, theta, N, r, s, P, p):  # checked
    """
    Compute the L matrix given the parameters
    Set size to be P (truncated)
    """
    L_MA = get_L_MA(lmbd, gamma, theta, N, r, s, P - p)
    L = np.zeros((P, p + r + 2 * s))
    L[:p, :p] = np.identity(p)
    L[p:, p:] = L_MA
    return L


def get_G(A, L):
    """
    Restore G from A and L
    G = A inv(L'L)L'
    """
    factor = np.matmul(np.linalg.inv(np.matmul(L.T, L)), L.T)
    G = mode_dot(A.numpy(), factor, 2)
    return G


def get_S_and_U(G, r1, r2, d):  # checked
    """
    Use HOOI to get S, U1 and U2 from G
    """
    S, U = tucker(G, rank=[r1, r2, d])
    S = mode_dot(S, U[2], 2)
    return (S, U[0], U[1])


def loss(y, A, T):
    summand = 0
    for t in range(1, T):  # starting from y_2
        y_temp = np.copy(y[:, :t])
        y_temp = np.flip(y_temp, axis=1)
        y_temp = np.reshape(y_temp, (-1, 1), order='F')
        A_temp = np.copy(A[:, :, :t])
        # a = (tensor_op.unfold(mode_dot(G, L_temp, 2), 1) @ y_temp).T[0]  # check later
        a = (tensor_op.unfold(A_temp, 1) @ y_temp)
        # a = 0
        # for j in range(t):
        #     for k in range(5):
        #         a = a + L[j,k] * G[:,:,k] @ y[:,t-j]
        summand = summand + np.linalg.norm(y[:, t:t+1] - np.array(a), ord=2) ** 2
    return summand / T


def loss_vec(Y, X1, G, L, T):
    A = mode_dot(G, L, 2)
    return sum(np.linalg.norm(Y - tensor_op.unfold(A[:, :, :T - 1], 1).numpy() @ X1, ord=2, axis=0) ** 2) / T


###############
# Derivatives #
###############

"""
Prepare calleble objective, Jacobian and Hessian functions
"""


def vec_jac_hess_lmbd(N, lmbd_k, k, G, L, y, Y, X1, p, T):
    """
    vectorization 
    assume already have 
    Y = [y1,...,yT] and 
    X1 = [y[:,:1],...,y[:,:T]] and
    X2 = 
    """

    X2 = np.zeros((N, T - 1, T - 1))  # can reduce dimension
    for i in range(p, T - 1):
        X2[:, i, :(i - p + 1)] = np.flip(y[:, :i - p + 1], axis=1)

    L_temp = np.copy(L[:(T - 1), :])
    L_temp[:, p + k] = 0
    a = Y - (tensor_op.unfold(mode_dot(G, L_temp, 2), 1).numpy() @ X1)  # N by (T-1)

    power_series = np.arange(1, T)
    lmbd_power = np.power(lmbd_k, power_series)  # find the power series of lmbd_k
    lmbd_y = np.einsum('k,jik->jik', lmbd_power, X2)
    lmbd_y_1 = np.einsum('k,ijk->ijk', power_series, lmbd_y / lmbd_k)
    lmbd_y_2 = np.einsum('k,ijk->ijk', power_series - 1, lmbd_y_1 / lmbd_k)
    outer_grad = a - G[:, :, p + k] @ np.sum(lmbd_y, axis=2)
    inner_grad = -G[:, :, p + k] @ np.sum(lmbd_y_1, axis=2)
    
    summand_j = 2 * np.einsum('ij,ij->', outer_grad, inner_grad)
    
    return summand_j / T


def vec_jac_hess_gamma_theta(N, eta_k, k, G, L, y, Y, X1, p, r, T):
    """ 
    Calculate the hessian matrix of (gamma,theta) pair
    """
    gamma_k = eta_k[0]
    theta_k = eta_k[1]

    X2 = np.zeros((N, T - 1, T - 1))
    for i in range(p+r, T - 1):
        X2[:, i, :(i - p-r + 1)] = np.flip(y[:, :i - p-r + 1], axis=1)

    L_temp = np.copy(L[:(T - 1), :])
    L_temp[:, p + r + k:p + r + k + 2] = 0  # set gamma_k and theta_k = 0 in L
    a = Y - (tensor_op.unfold(mode_dot(G, L_temp, 2), 1).numpy() @ X1)  # check later
    power_series = np.arange(1, T)
    gamma_power = np.power(gamma_k, power_series)

    cos_part = np.einsum('k,ijk,k->ijk', np.cos(theta_k * power_series), X2, gamma_power)  # gamma^(j-p) cos{(j-p)theta} y_{t-j}
    sin_part = np.einsum('k,ijk,k->ijk', np.sin(theta_k * power_series), X2, gamma_power)  # gamma^(j-p) sin{(j-p)theta} y_{t-j}
    cos_part_1 = np.einsum('k,ijk->ijk', power_series, (cos_part / gamma_k))  # (j-p) gamma^(j-p-1) cos{(j-p)theta} y_{t-j}
    sin_part_1 = np.einsum('k,ijk->ijk', power_series, (sin_part / gamma_k))  # (j-p) gamma^(j-p-1) sin{(j-p)theta} y_{t-j}
    
    A = G[:, :, p + r + 2 * k]
    B = G[:, :, p + r + 2 * k + 1]
    outer_grad = a - A @ np.sum(cos_part, axis=2) - B @ np.sum(sin_part, axis=2)
    inner_grad_gamma = -A @ np.sum(cos_part_1, axis=2) - B @ np.sum(sin_part_1, axis=2)
    inner_grad_theta = A @ np.einsum('k,ijk->ij', power_series, (sin_part)) - B @ np.einsum('k,ijk->ij', power_series, (cos_part))
    summand_gamma = 2 * np.einsum('ij,ij->', outer_grad, inner_grad_gamma)
    summand_theta = 2 * np.einsum('ij,ij->', outer_grad, inner_grad_theta)

    
    return summand_gamma / T, summand_theta / T


def soft_threshold(alpha, beta):
    if beta > alpha:
        return beta - alpha
    elif beta < -alpha:
        return beta + alpha
    else:
        return 0

def lasso_grd(N, i, j, k, G, L, Y, X1, T):
    L_temp = np.copy(L[:(T - 1), :])
    G_temp = np.copy(G)
    G_temp[i, j, k] = 0
    a = Y - (tensor_op.unfold(mode_dot(G_temp, L_temp, 2), 1).numpy() @ X1)
    
    coeff_ijk = np.zeros(T - 1)
    for cou in range(T - 1):
        coeff_ijk[cou] = L_temp[:, k].T @ X1[j::N, cou]
        
    outer_grad = a[i,:] - G[i,j,k]*coeff_ijk
    # summand_G1 = -2*outer_grad.T@coeff_ijk
    # summand_G2 = 2*sum(coeff_ijk)
    
    summand_G1 = -2*outer_grad.T@coeff_ijk
    # summand_G2 = coeff_ijk.T@coeff_ijk
    
    return summand_G1/T

# rsg = (r,s G, N, P, p, y, T)

def loss_lgt(lgt, r,s, G, N, P, p, y, T):
    #lmbd = lgt[0:rsg[0]]
    #gamma = lgt[rsg[0]:(rsg[0]+rsg[1])]
    #theta = lgt[(rsg[0]+rsg[1]):(rsg[0]+2*rsg[1])]
    
    lmbd = lgt[0:r]
    gamma = lgt[r:(r+s)]
    theta = lgt[(r+s):(r+2*s)]
    
    # get_L(lmbd, gamma, theta, N, r, s, P, p)
    # L = get_L(lmbd, gamma, theta, rsg[3], rsg[0], rsg[1], rsg[4], rsg[5])
    L = get_L(lmbd, gamma, theta, N, r, s, P, p)
    # loss(y, G, L, T)
    #resl = loss(rsg[6], rsg[2], L, rsg[7])
    resl = loss(y, G, L, T)
    
    return resl


'''
def lasso_update(i, j, k, lagr_mul, G, L, Y, X1, T, N):
    # G_up = G[i,j,k]
    L_temp = np.copy(L[:(T - 1), :])
    G_temp = np.copy(G)
    G_temp[i, j, k] = 0
    a = Y - (tensor_op.unfold(mode_dot(G_temp, L_temp, 2), 1).numpy() @ X1)
    # a = Y - (tensor_op.unfold(mode_dot(G, L,2),1).numpy() @ X1)

    coeff_ijk = np.zeros(T - 1)
    for cou in range(T - 1):
        coeff_ijk[cou] = L_temp[:, k].T @ X1[j::N, cou]

    # a_col = a.reshape((-1,1))
    a_col = a.reshape((-1, 1), order="F")[i::N]
    # coeff_ijk_full = np.zeros(N*(T-1))
    # coeff_ijk_full[i::N] = coeff_ijk
    inner_prod = coeff_ijk.T @ a_col
    new_G_up = soft_threshold(lagr_mul, inner_prod / T)

    return new_G_up
'''

#####################
# Useful Components #
#####################

"""
Compute components used in the ALS algorithm
z:
Z:
G_mat:
w_minus: 
oo
"""


def get_z(x, N, L):
    # !!!! need to check dimensions
    z = kron([L.T, np.identity(N)]) @ x
    return z


def get_sample_size(N, p, r, s):
    ratio = np.array([0.4, 0.3, 0.2, 0.1])
    d = r + 2 * s
    NofP = 4 * N + 4 * (p + d) + d
    ss = NofP / ratio
    return np.array(np.round(ss), dtype=int)
