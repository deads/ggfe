# Grammar-Guided Feature Extraction (GGFE)
#
# Author:   Damian Eads
#
# File:     image_features.py
#
# Purpose:  This module contains image processing primitives and
#           a grammar for generating signal processing programs for
#           feature extraction.
#
# Date:     January 13, 2010

import numpy as np
import scipy as sp
import scipy.ndimage as ndi
from core import variables, functions, Variable, Grammar, Environment
import scipy.stats

def smean(X):
    return X.mean(axis=1)

def var(X):
    return X.var(axis=1)
    
def stddev(X):
    return X.std(axis=1)

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

def erode(I, struct):
    "Performs a morphological erosion."
    return ndi.grey_erosion(I, footprint=struct)

def dilate(I, struct):
    "Performs a morphological dilation."
    return ndi.grey_dilation(I, footprint=struct)

def open(I, struct):
    "Performs a morphological opening."
    return ndi.grey_opening(I, footprint=struct)

def close(I, struct):
    "Performs a morpphological closing."
    return ndi.grey_closing(I, footprint=struct)

def kurtosis(X):
    return sp.stats.kurtosis(X, axis=1)

def window(X, a, b):
    n = X.shape[1]
    if b < a:
        a, b = swap(a, b)
    a = np.floor(n * a)
    b = np.floor(n * b)
    Xw = X[:,a:b]
    return Xw

def trapz(X):
    return np.trapz(X)

def swap(a, b):
    return b, a

def ptile(X, percentile, struct):
    "Computes the kth percentile filter with a particular structuring element."
    return ndi.percentile_filter(X, percentile, footprint=struct)

def line_selem(n):
    return np.ones((1, n))

def rand1n(n):
    "Generates a random integer between 1 and n (exclusive)."
    return np.random.randint(n)+1

def rand0n(n):
    "Generates a random integer between 0 and n (exclusive)."
    if n == 0:
        return 0
    else:
        return np.random.randint(n)

def sigmoid(I, a, b):
    "Soft maximum with shape parameters a and b."
    return  np.arctan(b * (I + a)) / b

def normDiff(X, Y):
    "Takes the normalized difference, i.e. r=(a-b)/(a+b)."
    sum = X + Y
    result = (X - Y) / sum
    result[sum == 0.0] = 0.0
    return result

def mult(X, Y):
    "Multiplies two images, i.e. r=a*b."
    return X * Y

def scaledSub(X, Y):
    "Takes the difference and scales, i.e. r=abs((a-b)/2+0.5)."
    result = (X - Y) / 2.0 + 0.5
    result[result < 0.0] = 0.0
    return result

def blend(*args):
    "Takes the average of several images, i.e. r=(a_1+a_2+...+a_n)/n"
    result = args[0]
    for arg in args[1:]:
        result = result + arg
    return result / float(len(args))

def get_signal_grammar():
    import signal_features
    reload(signal_features)
    grammar = Grammar("main_sig")


    Feature, SignalFeature, Morph, Compound, RandomSE, Statistic, Window = \
       grammar.productions(
        ["Feature", "SignalFeature", "Morph", "Compound", "RandomSE", "Statistic", "Window"]
        )

    X, Y, S = variables(["X", "Y", "S"])

    Binary, Unary, NLBinary, LBinary = \
        grammar.productions(
        ["Binary", "Unary", "NLBinary", "LBinary"]
        )
    erode, dilate, open, close, sigmoid, ptile = \
          functions(["erode", "dilate", "open", "close", "sigmoid", "ptile"],
                     module=signal_features)

    smean, stddev, var, skewness, kurtosis, trapz = \
        functions(["smean", "stddev", "var", "skewness", "kurtosis", "trapz"], module=signal_features);

    mult, normDiff, scaledSub, blend, line_selem, rand1n, window = \
          functions(["mult", "normDiff", "scaledSub", "blend", "line_selem", "rand1n", "window"],
                    module=signal_features)

    normal, random, randint, transpose, dot = \
            functions(["normal", "random", "randint", "transpose", "dot"],
                      module=np.random)

    [nan_to_num] = functions(["nan_to_num"])

    Feature[X] = (Statistic[SignalFeature[Window[X]]]
                  | nan_to_num(Statistic[SignalFeature[Window[X]]] /
                               Statistic[SignalFeature[Window[X]]]))

    SignalFeature[X] = (Binary[Unary[X], Unary[X]]
                        | Unary[X]
                        | Unary[Unary[X]]
                        | NLBinary[Unary[X], Unary[X]]
                        | Compound[X])

    Statistic[X] = (smean(X)
                    | stddev(X)
                    | var(X)
                    | skewness(X)
                    | kurtosis(X)
                    | trapz(X))

    Binary[X, Y] = (mult(X, Y)
                   | normDiff(X, Y)
                   | scaledSub(X, Y)
                   | blend(X, Y))

    Unary[X] = (sigmoid(X, ~normal(), 10 ** (~random() * 2 - 1))
                  | ptile(X, ~(random() * 100), RandomSE[...])
                  | Morph[X, RandomSE[...]])


    Morph[X,S] = (erode(X, S) |
                  dilate(X, S) |
                  open(X, S) |
                  close(X, S))
    
    Compound[X] = (Unary[X]
                   | Binary[X, Compound[X]])

    NLBinary[X, Y] = (mult(X, Y)
                      | normDiff(X, Y))
    
    LBinary[X, Y] = (scaledSub(X, Y)
                     | blend(X, Y))

    Window[X] = X | window(X, ~random(), ~random())

    RandomSE[...] = line_selem(rand1n(100))
    return grammar

def generate_and_evaluate(grammar, SIG):
    """
    Generate a random feature from the grammar passed and evaluate it
    on the image ``SIG``.
    """
    from numpy import pi, sin, cos, dot, transpose, nan_to_num
    feature = grammar.Feature(Variable('SIG'))
    print 'Evaluating random feature: ', str(feature)
    return eval(str(feature))

def eval_feature(program_string, SIG):
    from numpy import pi, sin, cos, dot, transpose, nan_to_num
    return eval(program_string)

def eval_stump(X, thresh, dir):
    if dir:
        return X >= thresh
    else:
        return X < thresh

def compute_werror(Yh, Y, Dt):
    return np.dot(Yh == Y, Dt).sum()

def dstump_find(X, Y, Dt):
    n = X.shape
    uX = np.unique(X)
    best_werror = np.dot(Y == 0, Dt)
    best_thresh = np.inf
    best_dir = True
    best_prev_thresh = None
    prev_thresh = None
    nT = uX.shape[0]
    for i in xrange(0, nT):
        thresh = uX[i]
        #print thresh
        if i+1 < nT:
            next_thresh = uX[i+1]
        else:
            next_thresh = None
        preds = eval_stump(X, thresh, True)
        werror = np.dot(preds != Y, Dt).sum()
        if werror < best_werror:
            best_thresh = thresh
            best_prev_thresh = prev_thresh
            best_werror = werror
            best_dir = True
        preds = eval_stump(X, thresh, False)
        werror = np.dot(preds != Y, Dt).sum()
        if werror < best_werror:
            best_thresh = thresh
            best_prev_thresh = next_thresh
            best_werror = werror
            best_dir = False
        prev_thresh = thresh
    if best_prev_thresh is not None:
        best_thresh = (best_thresh + best_prev_thresh) / 2.0
        if best_dir:
            best_werror = np.dot((X >= best_thresh) != Y, Dt).sum()
        else:
            best_werror = np.dot((X < best_thresh) != Y, Dt).sum()
    hyp = {}
    hyp["thresh"] = best_thresh
    hyp["dir"] = best_dir
    hyp["werror"] = best_werror
    return hyp

def get_alpha(werror):
    return .5 * np.log((1. - werror) / werror)

def adaboost_learn(grammar, X, Y, iterations, features_per_iteration):
    hypotheses = []
    n, m = X.shape
    Dt = np.ones((n,)) / n
    for it in xrange(0, iterations):
        best_werror = 1.0
        best_hypothesis = None
        best_result = None
        fnum = 0
        while best_werror >= 0.5 or fnum < features_per_iteration:
            feature = str(grammar.Feature(Variable('SIG')))
            result = eval_feature(feature, X)
            whyp = dstump_find(result, Y, Dt)
            print "It %d Feature %d Program %s WError: %5.8f Best WError: %5.8f" % (it, fnum, feature, whyp["werror"], best_werror)
            if whyp["werror"] < best_werror:
                best_werror = whyp["werror"]
                alpha = get_alpha(best_werror)
                best_hypothesis = whyp
                whyp["alpha"] = alpha
                best_result = result
            fnum += 1
        hypotheses.append(best_hypothesis)
        Yh = eval_stump(best_result, best_hypothesis["thresh"], best_hypothesis["dir"])
        Dt[(Yh == Y).ravel()] *= np.exp(-best_hypothesis["alpha"])
        Dt[(Yh != Y).ravel()] *= np.exp(best_hypothesis["alpha"])
        Dt /= Dt.sum()
        
    return hypotheses
            
