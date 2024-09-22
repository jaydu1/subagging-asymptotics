import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, ElasticNet, QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels
import scipy as sp
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from scipy.optimize import bisect, fixed_point
from scipy.stats import norm


from joblib import Parallel, delayed
from functools import reduce
import itertools

import warnings
warnings.filterwarnings('ignore') 


############################################################################
#
# Theoretical evaluation of ridge ensembles
#
############################################################################

# isotopic features

def v_phi_lam(phi, lam, a=1):
    '''
    The unique solution v for fixed-point equation
        1 / v(-lam;phi) = lam + phi * int r / 1 + r * v(-lam;phi) dH(r)
    where H is the distribution of eigenvalues of Sigma.
    For isotopic features Sigma = a*I, the solution has a closed form, which reads that
        lam>0:
            v(-lam;phi) = (-(phi+lam/a-1)+np.sqrt((phi+lam/a-1)**2+4*lam/a))/(2*lam)
        lam=0, phi>1
            v(-lam;phi) = 1/(a*(phi-1))
    and undefined otherwise.
    '''
    assert a>0
    
    min_lam = -(1 - np.sqrt(phi))**2 * a
    if phi<=0. or lam<min_lam:
        raise ValueError("The input parameters should satisfy phi>0 and lam>=min_lam.")
    
    if phi==np.inf:
        return 0
    elif lam!=0:
        return (-(phi+lam/a-1)+np.sqrt((phi+lam/a-1)**2+4*lam/a))/(2*lam)
    elif phi<=1.:
        return np.inf
    else:
        return 1/(a*(phi-1))
    

def tv_phi_lam(phi, phi_s, lam, v=None):
    if v is None:
        v = v_phi_lam(phi_s,lam)
        
    if v==np.inf:
        return phi/(1-phi)
    elif lam==0. and phi>1:
        return phi/(phi_s**2 - phi)
    else:
        tmp = phi/(1+v)**2
        tv = tmp/(1/v**2 - tmp)
        return tv
    

def tc_phi_lam(phi_s, lam, v=None):
    if v is None:
        v = v_phi_lam(phi_s,lam)
    if v==np.inf:
        return 0.
    elif lam==0 and phi_s>1:
        return (phi_s - 1)**2/phi_s**2
    else:
        return 1/(1+v)**2


def comp_risk_ridge(rho, sigma, lam, phi, psi):
    sigma2 = sigma**2
    if psi == np.inf:
        return rho**2 + sigma2, rho**2 + sigma2
    else:
        v = v_phi_lam(psi,lam)
        tc = rho**2 * tc_phi_lam(psi, lam, v)        
        tv = tv_phi_lam(phi, psi, lam, v)
        tv_s = tv_phi_lam(psi, psi, lam, v)
            
        risk_1 = (1 + tv_s) * tc + (1 + tv_s) * sigma2
        risk_inf = (1 + tv) * tc + (1 + tv) * sigma2
        return risk_1, risk_inf

############################################################################
#
# Theoretical evaluation of Lasso
#
############################################################################

def soft_threshold(x, tau):
    return np.maximum(np.abs(x) - tau, 0.) * np.sign(x)


def F1(a, zeta, epsilon, delta, lam, nu_p):
    '''
    Function F1 in Equation (32a)

    For X ~ N(0,1) and Theta ~ sparse * P_{nu} + (1-sparse) * P_0,
        zeta = sqrt(c * delta) * nu / tau
    '''
    prob = epsilon * (norm.cdf(-a+zeta) + norm.cdf(-zeta-a)) + 2 * (1-epsilon) * norm.cdf(-a) 
    if lam <= 1e-6:
        return prob - delta
    else:
        return lam / np.sqrt(delta) + a * (nu_p / zeta) / delta * (prob - delta)


def F2(zeta, epsilon, delta, lam, nu_p, sigma):
    '''
    Function F2 in Equation (32b)

    For X ~ N(0,1) and Theta ~ epsilon * P_{nu} + (1-epsilon) * P_0,
        zeta = sqrt(c * delta) * nu / tau
        SNR = sparse * nu**2 / sigma**2.

    Parameters
    ----------
    zeta : float
        The ratio nu_p / tau.
    epsilon : float
        The probability of nonzero signals.
    delta : float
        The inverse data aspect ratio n/p.
    lam : float
        The regularization parameter.
    nu_p : float
        The signal strength.
    sigma : float
        The noise level.
    '''
    a_max = np.maximum(1000,1000*lam)
    try:
        a = bisect(F1, 0, a_max, (zeta, epsilon, delta, lam, nu_p))
    except:
        null_risk = epsilon * nu_p**2 + sigma**2
        zetap = nu_p / np.sqrt(null_risk)
        a = bisect(F1, 0, a_max, (zetap, epsilon, delta, lam, nu_p))
    
    f = (sigma**2 * zeta**2 / nu_p**2 - 1) * delta 
    f += epsilon * ((zeta-a) * norm.pdf(zeta+a) + (a**2+1) * norm.cdf(-zeta-a))
    f += epsilon * zeta**2 * (norm.cdf(a-zeta) - norm.cdf(-a-zeta))
    f += epsilon * ((-zeta-a) * norm.pdf(-zeta+a) + (a**2+1) * norm.cdf(zeta-a))
    f += 2 * (1-epsilon) * (-a * norm.pdf(a) + (a**2+1) * norm.cdf(-a))
    f = np.abs(f) + zeta
    return f


def F3(x, c, delta, tau, a, sigma, HH, nu, epsilon):

    eta0 = np.clip(x, 0,1)
    rho = c * eta0

    if a==0:
        eta = sigma**2 / tau**2 +  rho / delta
    else:
        rho_u = np.sqrt(1+rho)
        rho_l = np.sqrt(1-rho)
        Sigma_harf = 0.5*np.array([
            [rho_u+rho_l, rho_u-rho_l], [rho_u-rho_l, rho_u+rho_l]
        ])
        HH_tau= HH @ Sigma_harf
        eta = sigma**2 / tau**2 +  (epsilon * np.mean(
                (soft_threshold(nu/tau + HH_tau[:,0], a) - nu/tau)*
                (soft_threshold(nu/tau + HH_tau[:,1], a) - nu/tau)
            ) + (1 - epsilon) * np.mean(
                (soft_threshold(HH_tau[:,0], a))*(soft_threshold(HH_tau[:,1], a))
        )) / delta

    return eta


def compute_risk_lasso(phi, lam, epsilon, nu, sigma, c = 1, eta=1., tau=None):
    '''
    Compute the risk of the Lasso ensemble.

    Parameters
    ----------
    phi : float
        The ratio p/n.
    lam : float
        The regularization parameter.
    epsilon : float
        The probability of Gaussian features.
    nu : float
        The signal strength. For X ~ N(0,1) and Theta ~ sparse * P_{nu} + (1-sparse) * P_0,
        SNR = sparse * nu**2 / sigma**2.
    sigma : float
        The noise level.
    c : float
        The subsample ratio k/n.
    eta : float
        The ratio of Gaussian features.
    '''
    if c>1:
        return np.nan, np.nan, eta
    if np.isinf(phi) or c==0 or np.isinf(lam):
        return epsilon*nu**2+sigma**2, epsilon*nu**2+sigma**2
        
    delta = c / phi # The inverse data aspect ratio k/p
    psi = 1 / delta
    # lam = np.maximum(lam/np.sqrt(phi)/np.sqrt(psi), 1e-6)

    nu_p = nu * np.sqrt(delta)
    lam = np.maximum(lam, 1e-7)
    if lam<=1e-6 and psi<=1:
        if psi<1:
            tau2 = 1 / (1 - psi) * sigma**2
            tau = np.sqrt(tau2)
            a = 0
        elif psi==1:
            tau = tau2 = np.inf
            a = 0
    else:
        if tau is None or np.isinf(tau) or np.isnan(tau):
            tau = sigma
        zeta = nu_p/tau
        zeta = fixed_point(F2, zeta, args=(epsilon, delta, lam, nu_p, sigma))
        # sol = fsolve(func=F2, x0=nu_p/sigma, args=(epsilon, delta, lam, nu_p, sigma))        
        # print(sol)
        # zeta = sol[0]
        a = bisect(F1, 0, np.maximum(1000,1000*lam), (zeta, epsilon, delta, lam, nu_p))
        tau = nu_p / zeta #* np.sqrt(delta)
        tau2 = tau**2

    if c==1:
        xi2 = tau2
    else:
        if psi<=1 and lam<=1e-6:
            xi2 = 1 / (1 - phi) * sigma**2
            eta = xi2/tau2
        else:
            m = 5000000
            HH = np.random.normal(size=(m, 2))
            # x0 = np.sqrt(tau**2-sigma**2)

            try:
                eta = fixed_point(F3, eta, args=(c, delta, tau, a, sigma, HH, nu_p, epsilon))
                xi2 = eta * tau**2 - sigma**2
            except:
                # xi = fsolve(func=F3, x0=x0, args=(c, delta, tau, a, sigma, HH, nu_p, epsilon))[0]
                xi2 = np.nan
            xi2 = xi2 + sigma**2
            xi2 = np.clip(xi2, sigma**2, tau**2)

    R1 = tau2
    Rinf = xi2
    return R1, Rinf, eta
    






############################################################################
#
# Theoretical evaluation of Huber
#
############################################################################


def zeta(x):
    return np.clip(x, -1, 1)

def l2_norm(x):
    # x: nparray of length sample 
    # return \|x\|_{L2} = \E[x^2]^{1/2}
    return np.sqrt(np.mean(x**2))

def F_4_unknown(param, H, G, Z, Theta, lam, mu, delta):
    alpha, beta, kappa, nu = param[0], param[1], param[2], param[3]
    zeta_H_Theta = lam/nu * zeta((beta*H+nu*Theta)/lam)
    zeta_G_Z = zeta((alpha*G+Z)/(mu + kappa))
    #update
    alpha_new = l2_norm(zeta_H_Theta-beta*H/nu)
    beta_new = np.sqrt(delta) * l2_norm(zeta_G_Z)
    kappa_new = 1/nu - np.mean(zeta_H_Theta*H)/beta
    nu_new = delta * np.mean(zeta_G_Z * G)/alpha
    return np.array([alpha_new, beta_new, np.max([kappa_new, 0]), np.max([nu_new, 0])]) ## nonnegative constarint

def solve_4_unknown(Theta, Z, lam, mu, delta, iter=5000, eps=1e-9):
    np.random.seed(0)
    G = np.random.normal(loc=0, scale=1, size=len(Z))
    H = np.random.normal(loc=0, scale=1, size=len(Theta))
    
    param = np.array([0.5,0.5/np.sqrt(delta),0.5,0.5/delta]) #init
    for i in range(iter):
        param_new = F_4_unknown(param, H=H, G=G, Z=Z, Theta=Theta, lam=lam, mu=mu, delta=delta)
        diff = np.sum((param_new - param)**2)
        param=param_new # Update 
        if diff < eps:
            break
        if i == iter-1:
            print('run full iteration: diff={}'.format(diff))
    return {'alpha':param[0], 'beta':param[1], 'kappa':param[2], 'nu':param[3], 'lam':lam, 'mu':mu, 'delta':delta}




############################################################################
#
# Theoretical evaluation of l1-Huber
#
############################################################################

def correlation(x, y):
    return np.mean(x*y)/l2_norm(x)/l2_norm(y)


def F_loss(etaG, alpha, kappa, mu, c, G, G_, Z):
    G_tilde = etaG * G + np.sqrt(1-etaG**2) * G_
    zeta_GZ =  zeta((alpha*G+Z)/(mu + kappa))
    zeta_GZ_tilde =  zeta((alpha*G_tilde+Z)/(mu + kappa))
    return c * correlation(zeta_GZ, zeta_GZ_tilde)

def F_reg(etaH, beta, nu, lam, H, H_, Theta):
    H_tilde = etaH * H + np.sqrt(1-etaH**2) * H_
    zeta_HTheta = lam/nu * zeta((beta*H+nu*Theta)/lam)
    zeta_HTheta_tilde = lam/nu * zeta((beta*H_tilde+nu*Theta)/lam)
    return correlation (zeta_HTheta-beta * H/nu, zeta_HTheta_tilde -beta *H_tilde/nu)

def compute_risk_huber_l1(Theta, Z, lam, mu, delta, c, iter=5000, eps=1e-9): 
    param_dict = solve_4_unknown(Theta=Theta, Z=Z, lam=lam, mu=mu, delta=c*delta)
    alpha = param_dict['alpha']
    beta = param_dict['beta']
    kappa = param_dict['kappa']
    nu = param_dict['nu']
    
    np.random.seed(0)
    G = np.random.normal(loc=0, scale=1, size=len(Z))
    G_ = np.random.normal(loc=0, scale=1, size=len(Z))
    H = np.random.normal(loc=0, scale=1, size=len(Theta))
    H_ = np.random.normal(loc=0, scale=1, size=len(Theta))

    if c==1:
        etaG, etaH = 1, 1
    else:
        etaG = 0
        for i in range(iter):
            etaH = zeta(F_loss(etaG=etaG, alpha=alpha, kappa=kappa, mu=mu, c=c, G=G, G_=G_, Z=Z)) #outer zeta projects value onto [-1,1]
            etaG_new = zeta(F_reg(etaH=etaH, beta=beta, nu=nu, lam=lam, H=H, H_=H_, Theta=Theta))
            diff = np.sum((etaG - etaG_new)**2)
            etaG = etaG_new # Update 
            if diff < eps:
                break
            if i == iter-1:
                print('run full iteration of eta: diff={}'.format(diff))
        etaH = F_loss(etaG=etaG, alpha=alpha, kappa=kappa, mu=mu, c=c, G=G, G_=G_, Z=Z)
    return {'alpha':alpha, 'beta':beta, 'kappa':kappa, 'nu':nu, 'etaG':etaG, 'etaH':etaH, 
            'lam':lam, 'mu':mu, 'delta':delta, 'c':c,
            'risk_limit': alpha**2 * etaG
            }


    
    
############################################################################
#
# Empirical evaluation
#
############################################################################


class Ridgeless(object):
    def __init__(self):        
        pass
    def fit(self, X, Y):
        self.coef_ = sp.linalg.lstsq(X, Y, check_finite=False, lapack_driver='gelsy')[0].T
        return self    
    def predict(self, X_test):
        return X_test @ self.coef_.T

def fit_predict(X, Y, X_test, method, param):
    sqrt_k = np.sqrt(X.shape[0])
    if method=='tree':
        regressor = DecisionTreeRegressor(max_features=1./3, min_samples_split=5)#, splitter='random')
        regressor.fit(X, Y)
        Y_hat = regressor.predict(X_test)
    elif method=='NN':
        regressor = MLPRegressor(random_state=0, max_iter=500).fit(X, Y)
        Y_hat = regressor.predict(X_test)
    elif method=='kNN':
        regressor = KNeighborsRegressor().fit(X, Y)
        Y_hat = regressor.predict(X_test)
    elif method=='logistic':
        clf = LogisticRegression(
            random_state=0, fit_intercept=False, C=1/np.maximum(param,1e-6)
        ).fit(X, Y.astype(int))
        Y_hat = clf.predict_proba(X_test)[:,1].astype(float)
    else:
        if method=='kernelridge':
            lam, kernel = param['lam'], param['kernel']
            degree = 3 if 'degree' not in param.keys() else param['degree']
            regressor = KernelRidge(alpha=X.shape[0]/X.shape[1]*lam, 
                                    kernel=kernel, coef0=0., degree=degree)
#             regressor.fit(X/sqrt_k, Y/sqrt_k)
#             Y_hat = regressor.predict(X_test/sqrt_k) * sqrt_k
            regressor.fit(X, Y)
        elif method=='ElasticNet':
            lam_1, lam_2 = param
            lam_1 = np.maximum(lam_1,1e-6)
            alpha = lam_1 + lam_2
            l1_ratio = lam_1 / (lam_1 + lam_2)
            regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
            regressor.fit(X, Y)
        else:
            lam = param
            if method=='ridge':
                regressor = Ridgeless() if lam==0 else Ridge(alpha=lam, fit_intercept=False, solver='lsqr')
    #             regressor = Ridge(alpha=np.maximum(lam,1e-6), fit_intercept=False, solver='lsqr')
                regressor.fit(X/sqrt_k, Y/sqrt_k)
            else:
                # lam = lam/X.shape[0]
                regressor = Lasso(alpha=np.maximum(lam,1e-6), 
                    tol=1e-8, max_iter=10000, fit_intercept=False)
                regressor.fit(X*sqrt_k, Y*sqrt_k)
                # regressor.fit(X, Y)
        Y_hat = regressor.predict(X_test)
        
    if len(Y_hat.shape)<2:
        Y_hat = Y_hat[:,None]
    return Y_hat


def comp_empirical_risk(X, Y, X_test, Y_test, psi, method, param, 
                        M=2, data_val=None, replace=True, 
                        return_allM=False, return_pred_diff=False):
    n,p = X.shape
    Y_test = Y_test.reshape((-1,1))    
    if data_val is not None:
        X_val, Y_val = data_val
        Y_val = Y_val.reshape((-1,1))
        Y_hat = np.zeros((Y_test.shape[0]+Y_val.shape[0], M))
        X_eval = np.r_[X_val, X_test]
    else:
        Y_hat = np.zeros((Y_test.shape[0], M))
        X_eval = X_test
        
    if replace:
        k = int(p/psi)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
    
    if k==0:
        Y_hat = np.zeros((X_eval.shape[0], M))
    else:
        with Parallel(n_jobs=8, verbose=0) as parallel:
            res = parallel(
                delayed(fit_predict)(X[ids,:], Y[ids,:], X_eval, method, param)
                for ids in ids_list
            )
        Y_hat = np.concatenate(res, axis=-1)
    #     for j in range(M):
    #         ids = ids_list[j]
    #         Y_hat[:,j:j+1] = fit_predict(X[ids,:]/np.sqrt(len(ids)), Y[ids,:]/np.sqrt(len(ids)), X_eval, param)
        
    if return_allM:
        Y_hat = np.cumsum(Y_hat, axis=1) / np.arange(1,M+1)
        idM = np.arange(M)
    else:
        Y_hat = np.mean(Y_hat, axis=1, keepdims=True)
        idM = 0
        
    if return_pred_diff:
        risk_test = (Y_hat[-Y_test.shape[0]:,:]-Y_test)[:,idM]
    else:
        risk_test = np.mean((Y_hat[-Y_test.shape[0]:,:]-Y_test)**2, axis=0)[idM]
        
    if data_val is not None:
        risk_val = np.mean((Y_hat[:-Y_test.shape[0],:]-Y_val)**2, axis=0)[idM]
        return risk_val, risk_test
    else:
        return risk_test

    






def huber_lasso_reg(y, X, lam, mu):
    n, p = X.shape
    if mu == 0:
        reg = QuantileRegressor(quantile=0.5, alpha=0.5*lam/n, fit_intercept=False, solver='highs')
        theta_hat = reg.fit(X=X, y=y).coef_
    else:
        assert mu>0
        X_new = np.concatenate((X, lam * np.eye(n)), axis=1)
        reg = Lasso(alpha=mu*lam/n, fit_intercept=False)
        theta_hat = reg.fit(X=X_new, y=y).coef_[0:p]
    return theta_hat

def compute_risk(n, p, s, sigma, dof, lam, mu, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0, scale=1/np.sqrt(p), size=(n, p))
    theta = np.hstack([np.zeros(int(p*s)), np.ones(p-int(p*s))]) * rng.normal(loc=0, scale=sigma, size=p)
    z = rng.standard_t(df=dof, size=n)
    y = X@theta + z
    theta_hat = huber_lasso_reg(y=y, X=X, lam=lam, mu=mu)
    
    return {
            'risk': np.mean((theta_hat-theta)**2)
            }

def compute_correlation(k, n, p, s, sigma, dof, lam, mu, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(loc=0, scale=1/np.sqrt(p), size=(n, p))
    theta = np.hstack([np.zeros(int(p*s)), np.ones(p-int(p*s))]) * rng.normal(loc=0, scale=sigma, size=p)
    z = rng.standard_t(df=dof, size=n)
    y = A@theta + z

    intersect = int(k**2/n)
    I_1 = np.arange(k)
    I_2 = np.arange(k-intersect, 2*k-intersect)
    h_hat1 = huber_lasso_reg(y=y[I_1], X=A[I_1,:], lam=lam, mu=mu)-theta
    h_hat2 = huber_lasso_reg(y=y[I_2], X=A[I_2,:], lam=lam, mu=mu)-theta

    return {
            'correlation': correlation(h_hat1, h_hat2),
            'risk': h_hat1.T @ h_hat2/p
            }

def compute_risk_huber_l1_emp(k, n, p, s, sigma, dof, lam, mu, seed):
    res_1 = compute_risk(k, p, s, sigma, dof, lam, mu, seed)
    res_inf = compute_correlation(k, n, p, s, sigma, dof, lam, mu, seed)

    return {
        'phi':p/n,
        'psi':p/k,
        'lam':lam,
        'mu':mu,
        'seed':seed,
        'R1':res_1['risk'],
        'Rinf':res_inf['risk']
    }







from sklearn_ensemble_cv import Ensemble
from sklearn.linear_model import Lasso, QuantileRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class HuberL1Reg(RegressorMixin, BaseEstimator):
    '''
    L1-regularized Huber estimator is defined as
        argmin_beta sum_i loss(y_i - x_i^T beta; mu) + sum_j reg(beta_j)
    where
        loss(z; mu) = z^2/(2*mu) * 1{|z|<=mu} + (|z| - mu/2) * 1{|z|>mu}
        reg(z) = |z|
    '''
    def __init__(self, lam, mu):
        assert lam>0
        if mu == 0:
            self.reg = QuantileRegressor(quantile=0.5, alpha=0.5*lam, fit_intercept=False, solver='highs')
        else:
            assert mu>0
            self.reg = Lasso(alpha=mu*lam, tol=1e-8, max_iter=10000, fit_intercept=False)
        self.fit_intercept = False
        self.lam = lam # l1-regularization parameter
        self.mu = mu   # Huber loss parameter
        self.is_fitted_ = False

    def fit(self, X, y):
        n, p = X.shape
        if self.mu == 0:
            self.coef_ = self.reg.fit(X=X, y=y).coef_
        else:
            X_new = np.concatenate((X, self.lam * np.eye(n)), axis=1)            
            self.coef_ = self.reg.fit(X=X_new*np.sqrt(n), y=y*np.sqrt(n)).coef_[0:p]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return X @ self.coef_

    def eval(self, r):
        '''
            loss(r; mu) = r^2/(2*mu) * 1{|r|<=mu} + (|r| - mu/2) * 1{|r|>mu}
        '''
        return sp.special.huber(self.mu, r)/self.mu

    def deriv(self, r):
        '''
            loss'(r; mu) = r/mu * 1{|r|<=mu} + sign(r) * 1{|r|>mu}
        '''
        return np.where(np.abs(r)<self.mu, r/self.mu, np.sign(r))

    def dof(self, eps=1e-12):
        return np.sum(np.abs(self.coef_)>eps)
        
    def get_gcv_input(self, X, y):
        r = y.reshape(-1) - self.predict(X).reshape(-1)
        loss_p = self.deriv(r)
        loss_pp = np.abs(r)<self.mu
        dof = self.dof()
        tr_V = (np.sum(loss_pp) - dof)/ self.mu
        
        return r, loss_p, dof, tr_V

def compute_risk_gcv(k, n, p, s, sigma, dof, lam, mu, seed, M=50, noise_est='theory', type='full'):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0, scale=1/np.sqrt(p), size=(n, p))
    theta = np.hstack([np.zeros(int(p*s)), np.ones(p-int(p*s))]) * rng.normal(loc=0, scale=sigma, size=p)
    z = rng.standard_t(df=dof, size=n)
    y = X @ theta + z
    
    kwargs_regr = {'lam': lam, 'mu': mu}
    kwargs_ensemble = {'max_samples':int(k), 'bootstrap': False, 'n_jobs':-1}
    regr = Ensemble(estimator=HuberL1Reg(**kwargs_regr), n_estimators=M, **kwargs_ensemble)
    regr.fit(X, y)
    res_est = regr.compute_cgcv_estimate(X, y, type=type)

    if noise_est=='theory':
        noise_1 = noise_inf = dof/(dof-2)
    else:
        if type=='full':
            noise_1 = noise_inf = np.mean(z**2)
        else:
            ids_list = [np.sort(ids) for ids in regr.estimators_samples_]

            def _get_est_var(i,j):
                ids = np.intersect1d(ids_list[i], ids_list[j])
                if len(ids)>0:
                    return np.mean(z[ids]**2)
                else:
                    return np.nan
            noise_1 = np.nanmean([_get_est_var(i,i) for i in np.arange(M)])
            noise_inf = np.nanmean([_get_est_var(i,j) for i in np.arange(M) for j in np.arange(M) if i < j])

    return {
        'phi':p/n,
        'psi':p/k,
        'lam':lam,
        'mu':mu,
        'seed':seed,
        'R1':res_est[0] - noise_1,
        'Rinf':(2*res_est[1]-res_est[0]) - noise_inf
    }


