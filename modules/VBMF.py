
from __future__ import division

import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import minimize_scalar

def EVBMF(Y, sigma2=None, H=None):
    """Implementation of the analytical solution to Empirical Variational Bayes Matrix Factorization.

    This function can be used to calculate the analytical solution to empirical VBMF. 
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix factorization."

    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free energy.
        If H is unspecified, it is set to the smallest of the sides of the input Y.

    Attributes
    ----------
    Y : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.
    
    sigma2 : int or None (default=None)
        Variance of the noise on Y.
        
    H : int or None (default = None)
        Maximum rank of the factorized matrices.
        
    Returns
    -------
    S : numpy-array
        Diagonal matrix of singular values.
        
    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.
    
    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.     
    """   
    L,M = Y.shape #has to be L<=M

    print(f"Y: {Y}")

    if H is None:
        H = L

    print(f"L: {L}")
    print(f"M: {M}")
    print(f"H: {H}")

    alpha = L/M
    tauubar = 2.5129*np.sqrt(alpha)

    print(f"alpha: {alpha}")
    print(f"tauubar: {tauubar}")
    
    #SVD of the input matrix, max rank of H
    _,s,_ = np.linalg.svd(Y)
    s = s[:H]

    print(f"s: {s}")

    #Calculate residual
    residual = 0.
    if H<L:
        residual = np.sum(np.sum(Y**2)-np.sum(s**2))

    #Estimation of the variance when sigma2 is unspecified
    if sigma2 is None: 
        xubar = (1+tauubar)*(1+alpha/tauubar)
        print(f"xubar: {xubar}")
        eH_ub = int(np.min([np.ceil(L/(1+alpha))-1, H]))-1
        print(f"eH_ub: {eH_ub}")
        upper_bound = (np.sum(s**2)+residual)/(L*M)
        lower_bound = np.max([s[eH_ub+1]**2/(M*xubar), np.mean(s[eH_ub+1:]**2)/M])
        print(f"upper_bound: {upper_bound}")
        print(f"lower_bound: {lower_bound}")

        scale = 1.#/lower_bound
        s = s*np.sqrt(scale)
        print(f"s: {s}")
        residual = residual*scale
        lower_bound = lower_bound*scale
        upper_bound = upper_bound*scale
        print(f"upper_bound: {upper_bound}")
        print(f"lower_bound: {lower_bound}")

        sigma2_opt = minimize_scalar(EVBsigma2, args=(L,M,s,residual,xubar), bounds=[lower_bound, upper_bound], method='Bounded')
        print(f"sigma2_opt: {sigma2_opt}")
        sigma2 = sigma2_opt.x
        print(f"sigma2: {sigma2}")

    #Threshold gamma term
    print(f"M: {M}")
    print(f"M*sigma2: {M*sigma2}")
    print(f"1+tauubar: {1+tauubar}")
    print(f"alpha/tauubar: {alpha/tauubar}")
    print(f"1+alpha/tauubar: {1+alpha/tauubar}")
    print(f"M*sigma2*(1+tauubar): {M*sigma2*(1+tauubar)}")
    print(f"M*sigma2*(1+tauubar)*(1+alpha/tauubar): {M*sigma2*(1+tauubar)*(1+alpha/tauubar)}")
    threshold = np.sqrt(M*sigma2*(1+tauubar)*(1+alpha/tauubar))
    print(f"threshold: {threshold}")
    pos = np.sum(s>threshold)

    #Formula (15) from [2]
    d = np.multiply(s[:pos]/2, 1-np.divide((L+M)*sigma2, s[:pos]**2) + np.sqrt((1-np.divide((L+M)*sigma2, s[:pos]**2))**2 -4*L*M*sigma2**2/s[:pos]**4) )
    return np.diag(d)

def EVBsigma2(sigma2,L,M,s,residual,xubar):
    H = len(s)

    alpha = L/M
    x = s**2/(M*sigma2) 

    z1 = x[x>xubar]
    z2 = x[x<=xubar]
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum( np.log( np.divide(tau_z1+1, z1)))
    term4 = alpha*np.sum(np.log(tau_z1/alpha+1))
    
    obj = term1+term2+term3+term4+ residual/(M*sigma2) + (L-H)*np.log(sigma2)

    return obj

def tau(x, alpha):
    return 0.5 * (x-(1+alpha) + np.sqrt((x-(1+alpha))**2 - 4*alpha))




