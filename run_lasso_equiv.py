import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import os
import sys
from compute_risk import compute_risk_lasso


s_list = [0.2,0.5,0.9]
sigma_list = [0.1, 0.5, 1., 1.5, 2., 5.]
sparse = s_list[int(sys.argv[1])]
sigma = sigma_list[int(sys.argv[2])]

phi = .1; psi_list = np.logspace(-1, 0.5, 100); lam_list = np.append([0, ], np.logspace(-2.,0.,100))
# phi = 1.1; psi_list = np.logspace(0, .7, 100); lam_list = np.append([0, ], np.logspace(-2.,1.,100))
psi_list = psi_list[psi_list>=phi]
psi_list = np.unique(psi_list)
lam_list = np.unique(lam_list)

path_result = 'result/lasso/equiv/{:.02f}_{:.02f}/'.format(sparse, sigma)
os.makedirs(path_result, exist_ok=True)


def run(phi, psi, sparse, sigma):
    risk_list = []
    eta = 1.

    for lam in tqdm(lam_list, desc='lam'):
        nu = 1 / np.sqrt(sparse) / sigma
        try:
            tau2, xi2, eta = compute_risk_lasso(
                phi, lam, sparse, nu, 1., c=phi/psi, eta=eta)
        except:
            tau2, xi2 = np.nan, np.nan
            print(phi,lam)
        risk_list.append([phi, psi, lam, sparse, nu, 1., tau2, xi2])
    risk_list = np.array(risk_list)
    return risk_list


with Parallel(n_jobs=16, verbose=12, timeout=99999) as parallel:
    res = parallel(delayed(run)(
        phi, psi, sparse=sparse, sigma=sigma) for psi in psi_list)
res = np.concatenate(res, axis=0)
df = pd.DataFrame(res, columns=['phi', 'psi', 'lam', 'sparse', 'nu', 'sigma', 'R1', 'Rinf'])
df.to_pickle(path_result+'res_lasso_risk_{:.01f}.pkl'.format(phi), compression='gzip')
