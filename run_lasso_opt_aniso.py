import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import os
import sys
from compute_risk import comp_risk_ridge, compute_risk_lasso, compute_risk_lasso_aniso


s_list = [0.01,0.5,0.9]
rho_list = [0.5, 1., 2., 5.]
rho_ar1_list = [0., 0.2, 0.4, 0.6]
sparse = s_list[int(sys.argv[1])]
rho = rho_list[int(sys.argv[2])]
rho_ar1 = rho_ar1_list[int(sys.argv[3])]
sigma = 1

phi_list = np.unique((np.round(np.logspace(-1, 1, 50), 2)))

path_result = 'result/lasso/opt_aniso/{:.02f}_{:.02f}_{:.02f}/'.format(sparse, rho, rho_ar1)
os.makedirs(path_result, exist_ok=True)

def ar1_cov(p, rho=0.5):
    idx = np.arange(p)
    Sigma = rho ** np.abs(np.subtract.outer(idx, idx))
    return Sigma

def matrix_sqrt_symmetric(A):
    """
    Compute the symmetric matrix square root of a symmetric positive definite matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(A)  # eigen-decomposition
    eigvals_sqrt = np.sqrt(np.maximum(eigvals, 0))  # numerical stability
    A_sqrt = eigvecs @ np.diag(eigvals_sqrt) @ eigvecs.T
    return A_sqrt

def run(phi, psi, lam, sparse, sigma):
    risk_list = []
    p = 200

    theta = np.ones(p) 
    theta[:int((1-sparse)*p)]=0
    theta *= rho/np.sqrt(sparse)

    cov = ar1_cov(p, rho_ar1) ## if rho=0 then covariance is isotropic
    cov_sqrt = matrix_sqrt_symmetric(cov)
    # print("\nCheck: Sigma_sqrt @ Sigma_sqrt.T ≈ Sigma ?", np.allclose(cov, cov_sqrt @ cov_sqrt.T))
    cov_sqrt_inv = np.linalg.inv(cov_sqrt)

    nu = rho / np.sqrt(sparse) / sigma
    try:
        tau2, xi2, eta = compute_risk_lasso_aniso(
            theta, cov_sqrt, cov_sqrt_inv, rho_ar1,
            phi, lam, sigma, c=phi/psi
        )
    except:
        tau2, xi2 = np.nan, np.nan
        print(phi,lam)
    risk_list.append([phi, psi, lam, sparse, nu, 1., tau2, xi2])
    risk_list = np.array(risk_list)
    return risk_list



with Parallel(n_jobs=16, verbose=12, timeout=99999) as parallel:
    lam = 0.
    for phi in tqdm(phi_list, desc='phi'):
        psi_list = np.r_[np.logspace(np.log10(np.maximum(1., phi)), 1, 200), np.logspace(1, 3, 100)]
        if phi<1:
            psi_list = np.r_[psi_list, np.logspace(np.log10(phi), 0, 10)]
        psi_list = np.round(psi_list, 2)        
        psi_list = np.unique(psi_list)
        psi_list = np.append(psi_list, [np.inf])
            
        res = parallel(delayed(run)(
            phi, psi, lam, sparse=sparse, sigma=sigma) for psi in tqdm(psi_list, desc='psi'))
        
        res = np.concatenate(res, axis=0)
        df = pd.DataFrame(res, columns=['phi', 'psi', 'lam', 'sparse', 'nu', 'sigma', 'R1', 'Rinf'])
        df.to_pickle(path_result+'res_lassoless_{:.02f}.pkl'.format(phi), compression='gzip')


    for phi in tqdm(phi_list, desc='phi'):
        psi = phi
        lam_list = np.r_[[0.], np.logspace(-3, 2, 300)]
        lam_list = np.round(lam_list, 3)        
        lam_list = np.unique(lam_list)
        lam_list = np.append(lam_list, [np.inf])

        res = parallel(delayed(run)(
            phi, psi, lam, sparse=sparse, sigma=sigma) for lam in tqdm(lam_list, desc='lam'))
        
        res = np.concatenate(res, axis=0)
        df = pd.DataFrame(res, columns=['phi', 'psi', 'lam', 'sparse', 'nu', 'sigma', 'R1', 'Rinf'])
        df.to_pickle(path_result+'res_lasso_{:.02f}.pkl'.format(phi), compression='gzip')