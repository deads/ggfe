#!/usr/bin/python

# Grammar-Guided Feature Extraction (GGFE)
#
# Author:   Damian Eads
#
# File:     image_features.py
#
# Purpose:  This module approximates correlations using numerical
#           approximations (w/ Gaussian quadrature) as opposed to
#           closed forms.
#
# Date:     May 26, 2010

import scipy.integrate as inte

import numpy as np

def gconic(d):
    "A inverted conic kernel."
    if d < 1.:
        return (1. - d)
    else:
        return 0.;

def gquad(d):
    "An inverted quadratic kernel."
    if d < 1.:
        return (1. - d ** 2.)
    else:
        return 0.;

def integrand_conic(d):
    "The integrand of a conic"
    return lambda x, y: gconic(np.sqrt((x - (d / 2.)) ** 2. + y ** 2.)) * \
                        gconic(np.sqrt((x + (d / 2.)) ** 2. + y ** 2.))

def integrand_quad(d):
    "The integrand of a quadratic"
    return lambda x, y: gquad(np.sqrt((x - (d / 2.)) ** 2. + y ** 2.)) * \
                        gquad(np.sqrt((x + (d / 2.)) ** 2. + y ** 2.))

def C(u, v, r, kernel='conic'):
    d = np.sqrt(np.dot(u - v, u - v)) / r
    beta = np.sqrt((4 - d ** 2.) / 2.)
    gfun = lambda x: 0.
    hfun = lambda x: (-d/2. - np.sqrt(1. - x ** 2.))
    print d
    if kernel == 'conic':
        integrand = integrand_conic(d)
    elif kernel == 'quad':
        integrand = integrand_quad(d)
    else:
        raise ValueError("Unknown kernel: " + kernel)
    s = inte.dblquad(integrand, -2., 2., lambda x: -2., lambda x: 2, ())
    return s

def create_cache(r, kernel):
    M = np.zeros((2*r+1, 2*r+1));
    u = np.array((0, 0))
    for i in xrange(0, 2*r-1):
        for j in xrange(i, 2*r-1):
            v = np.array((i, j), dtype=np.float)
            (M[i,j], err) = C(u, v, r, kernel);
            print '.',
    for i in xrange(0, 2*r-1):
        for j in xrange(0, i):
            M[i, j] = M[j, i]
    M /= M.max()
    return M
