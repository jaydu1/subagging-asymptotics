import os
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
sns.set()

from compute_risk import compute_risk_huber_l1

s = 0.1 # sparsity of signal
m = 100000
delta = 10



path_result = 'result/huber_l1/fixed_lam/'
os.path.exists(path_result) or os.makedirs(path_result)

cs = np.sort(np.unique((np.round(
    1/delta / np.logspace(-1, 1, 100), 2))))
lams = [1e-2,5e-2,1e-1,5e-1,1]
mus = np.linspace(10**(-3), 5, 100)

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
        Z = np.random.standard_cauchy(size=m) #* np.random.binomial(1, 0.8, size=m)
    Z *= sigma
    
    Theta = np.hstack([np.zeros(int(m*s)), np.ones(m-int(m*s))]) * np.random.normal(loc=0, scale=sigma, size=m)

    parameters = [dict(Theta=Theta, Z=Z, lam=lam, mu=mu, delta=delta, c=c)
            for lam in lams
            for mu in mus
            # for delta in [1.2]
            for c in cs
            ]

    print("Will now launch", len(parameters), "parallel tasks")
    data = Parallel(n_jobs=10, verbose=15)(delayed(compute_risk_huber_l1)(**d) for d in parameters)
    df = pd.DataFrame(data)
    df.to_csv(path_result+'res_{}_sigma_{}.csv'.format(dist, sigma))






path_result = 'result/huber_l1/fixed_mu/'
os.path.exists(path_result) or os.makedirs(path_result)

cs = np.sort(np.unique((
    1/delta / np.round(np.linspace(0.1, 2, 100), 2))))
lams = np.linspace(10**(-3), 1, 100)
mus = [1e-1,5e-1,1,5,10]

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
    Z *= sigma
    
    Theta = np.hstack([np.zeros(int(m*s)), np.ones(m-int(m*s))]) * np.random.normal(loc=0, scale=sigma, size=m)

    parameters = [dict(Theta=Theta, Z=Z, lam=lam, mu=mu, delta=delta, c=c)
            for lam in lams
            for mu in mus
            for c in cs
            ]

    print("Will now launch", len(parameters), "parallel tasks")
    data = Parallel(n_jobs=10, verbose=15)(delayed(compute_risk_huber_l1)(**d) for d in parameters)
    df = pd.DataFrame(data)
    df.to_csv(path_result+'res_{}_sigma_{}.csv'.format(dist, sigma))


