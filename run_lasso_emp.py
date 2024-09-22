import os
import sys


import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from generate_data import *
from compute_risk import *
from tqdm import tqdm

n_simu = 50

M = 100

s_list = [0.1,0.5,0.9]
sigma_list = [1., 2., 5.]
sparse = s_list[int(sys.argv[1])]
sigma = sigma_list[int(sys.argv[2])]
rho = sigma * 2

if int(sys.argv[3])==0:
    phi = .1; psi_list = np.logspace(-1., 1., 100); lam_list = np.array([0., 1e-5, 1e-4, 1e-3, 1e-2, 0.1]); n = 2000
else:
    phi = 1.1; psi_list = np.logspace(np.log10(phi), 1., 100); lam_list = np.array([0., 1e-5, 1e-4, 1e-3, 1e-2, 0.1]); n = 500
psi_list = np.round(psi_list, 2)
psi_list = psi_list[psi_list>=phi]
psi_list = np.unique(psi_list)
lam_list = np.unique(lam_list)

path_result = 'result/lasso/emp/{:.02f}_{:.02f}/'.format(sparse, sigma)
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(phi, psi, lam, M, i):
    p = int(n*phi)

    coef = 'sparse-{}'.format(int(p*sparse))
    if psi<phi:
        R = np.full(M, np.nan)
    else:
        np.random.seed(i)
        _, _, X, Y, X_test, Y_test, _, _ = generate_data(
            p, phi, 0., sigma, rho=rho, coef=coef, func='linear', n_test=1000)
        X, X_test = X / np.sqrt(p), X_test / np.sqrt(p)

        R = comp_empirical_risk(X, Y, X_test, Y_test, psi, 'lasso', lam, 
                            M=M, return_allM=True)
    return np.c_[
        np.tile(np.array([phi,psi,lam,i]), (M,1)), 1+np.arange(M), R
        ]


def run_theory(phi, psi, sparse, sigma):
    risk_list = []
    eta = 1.
    tau = None
    for lam in tqdm(lam_list, desc='lam'):
        nu = rho / np.sqrt(sparse) * sigma
        try:
            tau2, xi2, eta = compute_risk_lasso(
                phi, lam, sparse, nu, 1., c=phi/psi, eta=eta, tau=tau)
        except:
            tau2, xi2 = np.nan, np.nan
            print(phi, psi, lam, tau)
        risk_list.append([phi, psi, lam, sparse, nu, 1., tau2, xi2])

        if not np.isinf(tau2) and not np.isnan(tau2):            
            tau = np.sqrt(tau2)
    risk_list = np.array(risk_list)
    return risk_list


from joblib import Parallel, delayed
with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:            
    res = parallel(
        delayed(run_one_simulation)(phi, psi, lam, M, i)
            for psi in tqdm(psi_list, desc='psi') for lam in tqdm(lam_list, desc='lam') for i in tqdm(np.arange(n_simu), desc='i')
    )

    df = pd.DataFrame(np.concatenate(res, axis=0), columns=['phi','psi','lam','i','M','risk'])
    df.to_pickle(path_result+'res_est_phi_{:.01f}.pkl'.format(phi), compression='gzip')



    res = parallel(delayed(run_theory)(
        phi, psi, sparse=sparse, sigma=sigma) for psi in psi_list #for lam in tqdm(lam_list, desc='lam')
        )
    res = np.concatenate(res, axis=0)
    df = pd.DataFrame(res, columns=['phi', 'psi', 'lam', 'sparse', 'nu', 'sigma', 'R1', 'Rinf'])
    df.to_pickle(path_result+'res_risk_phi_{:.01f}.pkl'.format(phi), compression='gzip')
