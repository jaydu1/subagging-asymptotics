import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd


path_result = 'result/huber/'
import os
os.path.exists(path_result) or os.makedirs(path_result)


def zeta(x):
    return np.where(x>1., 1., np.where(x<-1., -1., x))

def kappa_band(alpha, delta, Lambda):
    gamma = 1/delta
    lb = np.minimum(gamma/(1-gamma), alpha*np.sqrt(gamma)/Lambda)
    ub = gamma/(2*(1-gamma)) * (1 + np.sqrt(1 + 4*(1-gamma)/(Lambda**2 * gamma) * alpha**2))
    return (lb, ub)

def alpha_kappa_eq(x, delta, Lambda, G, Z):
    alpha = x[0]
    kappa = x[1]
    xi = (alpha * G + Z)/Lambda/(1+kappa)
    F1 = alpha**2 - kappa**2 * Lambda**2 * delta * np.mean(zeta(xi)**2)
    F2 = alpha - kappa * Lambda * delta * np.mean(zeta(xi)*G)
    return np.array([F1, F2])

def eta_eq(eta, alpha, kappa, c, Lambda, GG, Z):
    a = np.sqrt(1+eta)
    b = np.sqrt(1-eta)
    Sigma_harf = 0.5*np.array([
        [a+b, a-b], [a-b, a+b]
    ])
    GG_eta= GG @ Sigma_harf
    xi = (alpha * GG_eta[:,0] + Z)/(Lambda*(1+kappa))
    xi_tilde = (alpha * GG_eta[:,1] + Z)/(Lambda*(1+kappa))
    return eta - c * np.mean(zeta(xi)*zeta(xi_tilde))/np.mean(zeta(xi)**2)

def solve(c, delta, Lambda, G, GG, Z, alpha_0):
    # alpha_0: start point of alpha
    l, u = kappa_band(alpha_0, delta*c, Lambda)
    kappa_0 = (l + u) / 2
    alpha, kappa = fsolve(func=alpha_kappa_eq, x0=np.array([alpha_0, kappa_0]), args=(delta*c, Lambda, G, Z))
    eta = root_scalar(f=eta_eq, args=(alpha, kappa, c, Lambda, GG, Z), x0=0, bracket=[-c, c]).root
    return np.array([alpha, kappa, eta])

def compute_equation(c, delta, lam, G, GG, Z, alpha_0):
    alpha, kappa, eta = solve(c=c, delta=delta, Lambda=lam, G=G, GG=GG, Z=Z, alpha_0=alpha_0)
    return {'alpha':alpha, 'kappa':kappa, 'eta':eta, 'c':c, 'lam':lam}


for dist in ['t-2','t-10','t-20']:

    if dist[0] == 't':
        dof = int(dist.split('-')[1])
    sigma = 1.

    np.random.seed(10)
    m = 100000
    G = np.random.normal(size=m)
    GG = np.random.normal(size=(m, 2))
    if dist == 'normal':
        Z = np.random.normal(size=m)
    elif dist[0] == 't':
        Z = np.random.standard_t(df=dof, size=m)
    elif dist == 'cauchy':
        Z = np.random.standard_cauchy(size=m)
    Z *= sigma
    delta = 5
    cs = np.linspace(1/delta+0.01, 1., 100)


    cs = np.sort(1/delta / np.logspace(np.log10(1/delta), np.log10(1/delta/0.21), 100))
    lams = np.linspace(10**(-3), 5, 100)

    params = [dict(c=c, delta=delta, lam=lam, G=G, GG=GG, Z=Z, alpha_0=5)
            for c in cs
            for lam in lams
            ]

    def compute_equation(c, delta, lam, G, GG, Z, alpha_0):
        alpha, kappa, eta = solve(c=c, delta=delta, Lambda=lam, G=G, GG=GG, Z=Z, alpha_0=alpha_0)
        return {'alpha':alpha, 'kappa':kappa, 'eta':eta, 'c':c, 'lam':lam}

    print("Will now launch", len(params), "parallel tasks")
    data_eq = Parallel(n_jobs=5, verbose=6)(delayed(compute_equation)(**d) for d in params)
    df_eq = pd.DataFrame(data_eq)
    df_eq.to_csv(path_result+'res_{}_sigma_{}.csv'.format(dist, sigma))
