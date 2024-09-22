import os
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
sns.set()

from compute_risk import compute_risk_huber_l1

s = 0.7 # sparsity of signal
m = 100000

path_result = 'result/huber_l1/opt/'
os.path.exists(path_result) or os.makedirs(path_result)


phi_list = np.unique((np.round(np.logspace(-1, 1, 50), 2)))

lams = [1e-1,5e-1,1]
mus = [5e-1,1,5]

for dist in ['t-2','t-10','t-20']:

    if dist[0] == 't':
        dof = int(dist.split('-')[1])
    sigma = 1.

    np.random.seed(10)
    G = np.random.normal(size=m)
    GG = np.random.normal(size=(m, 2))
    if dist == 'normal':
        Z = np.random.normal(size=m)
    elif dist[0] == 't':
        Z = np.random.standard_t(df=dof, size=m)
    elif dist == 'cauchy':
        Z = np.random.standard_cauchy(size=m)
    
    Theta = np.hstack([np.zeros(int(m*s)), np.ones(m-int(m*s))]) * np.random.normal(loc=0, scale=sigma, size=m)

    for phi in tqdm(phi_list, desc='phi'):
            
        psi_list = np.r_[np.logspace(np.log10(np.maximum(1., phi)), 1, 50), np.logspace(1, 3, 25)]
        if phi<1:
            psi_list = np.r_[psi_list, np.logspace(np.log10(phi), 0, 10)]
        psi_list = np.round(psi_list, 2)
        psi_list = np.unique(psi_list)
        cs = phi / psi_list
        parameters = [dict(Theta=Theta, Z=Z, lam=lam, mu=mu, delta=1/phi, c=c)
                for lam in lams
                for mu in mus
                for c in cs
                ]

        print("Will now launch", len(parameters), "parallel tasks")
        data = Parallel(n_jobs=10, verbose=15)(delayed(compute_risk_huber_l1)(**d) for d in parameters)
        df = pd.DataFrame(data)
        df.to_csv(path_result+'res_{}_sigma_{}_{:.02f}.csv'.format(dist, sigma, phi))




        
        