import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.linalg import sqrtm


def create_toeplitz_cov_mat(sigma_sq, first_column_except_1):
    '''
    Parameters
    ----------
    sigma_sq: float
        scalar_like,
    first_column_except_1: array
        1d-array, except diagonal 1.
    
    Return:
    ----------
        2d-array with dimension (len(first_column)+1, len(first_column)+1)
    '''
    first_column = np.append(1, first_column_except_1)
    cov_mat = sigma_sq * sp.linalg.toeplitz(first_column)
    return cov_mat


def ar1_cov(rho, n, sigma_sq=1):
    """
    Parameters
    ----------
    sigma_sq : float
        scalar
    rho : float
        scalar, should be within -1 and 1.
    n : int
    
    Return
    ----------
        2d-matrix of (n,n)
    """
    if rho!=0.:
        rho_tile = rho * np.ones([n - 1])
        first_column_except_1 = np.cumprod(rho_tile)
        cov_mat = create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
    else:
        cov_mat = np.identity(n)
    return cov_mat



def generate_data(p, phi, rho_ar1=1., sigma=1, coef='random',
                  func='quad', df=np.inf, 
                  rho=1., n_test=1000, Sigma=None, beta0=None):
    n = int(p/phi)
        
    if Sigma is None:
        Sigma = ar1_cov(rho_ar1, p)
    
#     if cov=='ar1':
        
#     elif cov=='random':
#         s = np.diag(np.random.uniform(1., 2., size=p))
#         Q, _ = np.linalg.qr(np.random.rand(p, p))
#         Sigma = Q.T @ s @ Q
    if df==np.inf:
        Z = np.random.normal(size=(n,p))
        Z_test = np.random.normal(size=(n_test,p))
    else:
        Z = np.random.standard_t(df=df, size=(n,p)) / np.sqrt(df / (df - 2))
        Z_test = np.random.standard_t(df=df, size=(n_test,p)) / np.sqrt(df / (df - 2))
    
    Sigma_sqrt = sqrtm(Sigma)
    X = Z @ Sigma_sqrt #/ np.sqrt(p)
    X_test = Z_test @ Sigma_sqrt #/ np.sqrt(p)
    
    if beta0 is None:
        if sigma<np.inf:
            if coef=='random':
                beta0 = np.random.normal(size=(p,))
                beta0 /= np.linalg.norm(beta0)                
                rho2 = 1.      
            elif coef.startswith('sparse'):
                s = int(coef.split('-')[1])
                beta0 = np.zeros(p,)
                # beta0[:s] = 1/np.sqrt(s)
                beta0 = np.random.binomial(1, s/p, size=p) / np.sqrt(s)
                rho2 = 1.
            elif coef.startswith('eig'):
                top_k = int(coef.split('-')[1])
                _, beta0 = eigsh(Sigma, k=top_k)
                rho2 = (1-rho_ar1**2)/(1-rho_ar1)**2/top_k
                beta0 = np.mean(beta0, axis=-1)
        else:
            rho2 = 0.
            beta0 = np.zeros(p)
        beta0 *= rho
        rho2 *= rho**2
    else:
        rho2 = None

    Y = X@beta0[:,None]   
    Y_test = X_test@beta0[:,None]
    
    if func=='linear':
        pass
    elif func=='tanh':
        Y = np.tanh(Y)
        Y_test = np.tanh(Y_test)
    elif func=='quad':
        Y += (np.mean(X**2, axis=-1) - np.trace(Sigma)/p)[:, None]
        Y_test += (np.mean(X_test**2, axis=-1) - np.trace(Sigma)/p)[:, None]
    else:
        raise ValueError('Not implemented.')
    
    if sigma>0.:
        if df==np.inf:
            Y += np.random.normal(size=(n,1))
            Y_test += np.random.normal(size=(n_test,1))
        else:
            Y += np.random.standard_t(df=df, size=(n,1))
            Y_test += np.random.standard_t(df=df, size=(n_test,1))
        sigma = 1. if df==np.inf else np.sqrt(df / (df - 2))
    else:
        sigma = 0.

    return Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma**2


