
import os
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
sns.set()

from compute_risk import compute_risk_huber_l1, compute_risk_huber_l1_emp



path_result = 'result/huber_l1/emp/'
os.path.exists(path_result) or os.makedirs(path_result)

s = 0.1 # sparsity of signal
sigma = 1
sample = 100000

lams = [0.1, 0.2, 0.5]
mus = [1, 5]
dist = 't-5'; noise_est='empirical'
# dist = 't-2'; noise_est='empirical'

delta = 10;phi=1/delta
cs = np.sort(np.unique((
    1/delta / np.round(np.logspace(-1, 1, 200), 2))))

np.random.seed(0)
Theta = np.hstack([np.zeros(int(sample*s)), np.ones(sample-int(sample*s))]) * np.random.normal(loc=0, scale=sigma, size=sample)

dof = int(dist.split('-')[1])
G = np.random.normal(size=sample)
GG = np.random.normal(size=(sample, 2))
Z = np.random.standard_t(df=dof, size=sample)

parameters = [dict(Theta=Theta, Z=Z, lam=lam, mu=mu, delta=delta, c=c)
        for lam in lams
        for mu in mus
        for c in cs
        ]

print("Will now launch", len(parameters), "parallel tasks")
data = Parallel(n_jobs=10, verbose=15)(delayed(compute_risk_huber_l1)(**d) for d in parameters)
df = pd.DataFrame(data)
df['phi'] = 1/df['delta']
df['psi'] = 1/(df['c']*df['delta'])
df['R1'] = df['alpha']**2
df['Rinf'] = df['risk_limit']
df = df[['phi', 'psi', 'lam', 'mu', 'R1', 'Rinf']]
df.to_pickle(path_result+'res_risk_{}_phi_{:.01f}.pkl'.format(dist,phi), compression='gzip')


cs = np.sort(np.unique((
    1/delta / np.round(np.logspace(-1, 1, 25), 2))))
cs = cs[1/(delta*cs)<=6.]
p = 500
n = int(delta*p)

params = [{'k': k, 'n':n, 'p':p, 's':s,'sigma':sigma,'dof':dof, 'lam':lam, 'mu':mu, 'seed': seed}
          for k in (cs*n).astype(int)
          for lam in lams
          for mu in mus
          for seed in range(50)
        ]

print('about to start parallel jobs:', len(params))
data_finite = Parallel(n_jobs=-1, verbose=15)(
        delayed(compute_risk_huber_l1_emp)(**d) for d in params)
df = pd.DataFrame(data_finite)
df.to_pickle(path_result+'res_est_{}_phi_{:.01f}.pkl'.format(dist,phi), compression='gzip')




from compute_risk import compute_risk_gcv

p = 500
n = int(delta*p)

params = [{'k': k, 'n':n, 'p':p, 's':s,'sigma':sigma,'dof':dof, 'lam':lam, 'mu':mu, 'seed': seed, 'noise_est':noise_est}
          for k in (cs*n).astype(int)
          for lam in lams
          for mu in mus
          for seed in range(50)
        ]
        
print('about to start parallel jobs:', len(params))
data_finite = Parallel(n_jobs=5, verbose=15)(
        delayed(compute_risk_gcv)(**d) for d in params)
df = pd.DataFrame(data_finite)
df.to_pickle(path_result+'res_gcv_full_{}_phi_{:.01f}.pkl'.format(dist,phi), compression='gzip')