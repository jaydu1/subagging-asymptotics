import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg

from generate_data import *
from compute_risk import *
from tqdm import tqdm


# methods and hyperparameters
method = 'lassoless'
print(method)

path_result = 'result/lassoless/'

rho = 1; n_simu = 100; sigma = 0.5


params_phi = np.logspace(np.log10(0.25), np.log10(2.5), 20)
params_phi = np.r_[params_phi, [1.]]
# params_n = (d / params_phi).astype(int)
lam = 0.
os.makedirs(path_result, exist_ok=True)
# eps = int(coef.split('-')[1])/d

eps = 0.2

def run_one_simulation_thm(lam, phi, sparse, sigma):
    risk_list = []
    nu = rho / np.sqrt(sparse)
    try:
        a, tau, err_R = compute_risk_lassoless(phi, lam, sparse, nu, sigma)
    except:
        a, tau, err_R = np.nan, np.nan, np.nan, np.nan, np.nan
        print(phi,lam)
    risk_list.append([phi, lam, sparse, nu, sigma, a, tau, err_R])
    risk_list = np.array(risk_list)
    return risk_list
    

with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:

    df_res = pd.DataFrame()

    res = parallel(
        delayed(run_one_simulation_thm)(lam, phi, eps, sigma) for phi in tqdm(params_phi, desc='phi')
    )    

    res = pd.DataFrame(np.concatenate(res,axis=0), columns=
        ['phi', 'lam', 'sparse', 'nu', 'sigma', 'a', 'tau', 'err_R']
    )
    df_res = pd.concat([df_res, res],axis=0)

    df_res.to_csv('{}res_risk_{}_eps_{:.02f}_sigma_{:.02f}.csv'.format(
        path_result, method, eps, sigma), index=False)

