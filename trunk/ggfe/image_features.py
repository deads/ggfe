import numpy as np
import scipy.ndimage as ndi

from numpy import matrix

from core import variables, functions, Variable, Grammar, Environment

def erode(I, struct):
    return ndi.grey_erosion(I, footprint=struct)

def dilate(I, struct):
    return ndi.grey_dilation(I, footprint=struct)

def open(I, struct):
    return ndi.grey_opening(I, footprint=struct)

def close(I, struct):
    return ndi.grey_closing(I, footprint=struct)

def normDiff(X, Y):
    """Takes the normalized difference, i.e. r=(a-b)/(a+b)."""
    sum = X + Y
    result = (X - Y) / sum
    result[sum == 0.0] = 0.0
    return result

def mult(X, Y):
    """Multiplies two images, i.e. r=a*b."""
    return X * Y

def scaledSub(X, Y):
    """Takes the difference and scales, i.e. r=abs((a-b)/2+0.5)."""
    result = (X - Y) / 2.0 + 0.5
    result[result < 0.0] = 0.0
    return result

def blend(*args):
    """Takes the average of several images, i.e. r=(a_1+a_2+...+a_n)/n"""
    result = args[0]
    for arg in args[1:]:
        result = result + arg
    return result / float(len(args))

L5 = np.array([[ 1,  4, 6,  4,  1]], dtype='float')
E5 = np.array([[-1, -2, 0,  2,  1]], dtype='float')
S5 = np.array([[-1,  0, 2,  0, -1]], dtype='float')
R5 = np.array([[ 1, -4, 6, -4,  1]], dtype='float')
W5 = np.array([[ -1, 2, 0, -2,  1]], dtype='float')
laws_base = [L5, E5, S5, R5, W5]

laws_masks = []
for i in xrange(0,len(laws_base)):
    for j in xrange(0,len(laws_base)):
        laws_masks.append(np.dot(laws_base[i].T, laws_base[j]))

def laws(X, i):
    return ndi.convolve(X, laws_masks[i])

def ptile(X, percentile, struct):
    return ndi.percentile_filter(X, percentile, footprint=struct)

def ggm(X, std):
    return ndi.gaussian_gradient_magnitude(X, std)

def laplace(X, sigma):
    """Applies a Laplace to an image."""
    return ndi.gaussian_laplace(X, sigma)

def sigmoid(I, a, b):
    return  np.arctan(b * (I + a)) / b

def selem(angle, size, ratio):
    structure = np.zeros((2*size+1, 2*size+1))
    
    a = matrix([np.cos(angle), np.sin(angle)])
    b = matrix([-np.sin(angle), np.cos(angle)])
    aspect = a.T * a +  b.T * b / ratio

    for y in xrange(-size, size+ 1):
        for x in xrange(-size, size + 1):
            i = x+size
            j = y+size
            
            v = matrix([x,y], dtype='f') / float(size)
            n = v * aspect * v.T

            if n < 1:
                structure[i, j] = 1

    return structure

def get_image_grammar():
    import image_features
    reload(image_features)
    grammar = Grammar("main")

    Feature, Morph, Compound, RandomSE = grammar.productions(["Feature", "Morph", "Compound", "RandomSE"])

    Binary, Unary = grammar.productions(["Binary", "Unary"])
    NLUnary, LUnary = grammar.productions(["NLUnary", "LUnary"])
    NLBinary, LBinary = grammar.productions(["NLBinary", "LBinary"])

    erode, dilate, open, close, sigmoid = functions(["erode", "dilate", "open", "close", "sigmoid"], module=image_features)
    mult, normDiff, scaledSub, blend = functions(["mult", "normDiff", "scaledSub", "blend"], module=image_features)

    laws, laplace, ptile, ggm, selem = functions(["laws", "laplace", "ptile", "ggm", "selem"], module=image_features)
    normal, random, randint = functions(["normal", "random", "randint"], module=np.random)

    X, Y, S, pi, k = variables(["X", "Y", "S", "pi", "k"])
    
    Feature[X] = (Binary[Unary[X], Unary[X]]
                  | NLUnary[Unary[X]]
                  | NLBinary[Unary[X], Unary[X]]
                  | Compound[X])
    
    Binary[X, Y] = (mult(X, Y)
                   | normDiff(X, Y)
                   | scaledSub(X, Y)
                   | blend(X, Y))

    Unary[X] = (LUnary[X]
                | NLUnary[X])

    NLUnary[X] = (sigmoid(X, ~normal(), 10**(~random()*2 - 1))
                  | ptile(X, ~(random() * 100), RandomSE[...])
                  | ggm(X, ~normal() * 3)
                  | Morph[X, RandomSE[...]])

    LUnary[X] = (laws(X, ~randint(25))
                 | laplace(X, ~normal() * 3))

    RandomSE[...] = selem((~random() * 3.14),
                          ((1+~randint(7))*2+1),
                          (10 ** (~random() * 2 - 1)))


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

    return grammar

def generate_and_run(grammar, IMG):
    feature = grammar.Feature(Variable('IMG'))
    
    print 'Evaluating random feature: ', str(feature)
    environment = Environment()
    environment.globals['IMG'] = IMG
    return eval(str(feature))

def generate_random_features(grammar, num):
    IMG = variables(['IMG'])
    features = []
    for i in xrange(0, num):
        feature = grammar.Feature(IMG)
        features.append(feature)
    return features
