import numpy as np

def smean(X):
    X.mean(axis=1)

def var(X):
    X.var(axis=1)
    
def stddev(X):
    X.std(axis=1)

def skewness(X):
    """
    deviation = (X - X.sum(axis=1))
    g1 = (deviation ** 3.0).mean(axis=1)
    g1 /= (deviation ** 2.0).mean(axis=1) ** (3./2.)
    n = X.shape[1]
    k3 = np.sqrt(n * (n - 1));
    k2_ = n - 2
    """
    return sp.stats.skew(X, axis=1)

def kurtosis(X):
    return sp.stats.kurtosis(X, axis=1)

def window(X, a, b):
    n = X.shape[1]
    a = np.floor(n * a)
    b = np.floor(n * b)
    Xw = X[:,a:b]
    return Xw

def trapz(X):
    return trapz(X)
    
