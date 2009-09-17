# Grammar-Guided Feature Extraction (GGFE)
#
# Author:   Damian Eads
#
# File:     image_features.py
#
# Purpose:  This module contains image processing primitives and
#           a grammar for generating image processing programs for
#           feature extraction.
#
# Date:     August 8, 2009

import numpy as np
import scipy.ndimage as ndi

from numpy import matrix

from core import variables, functions, Variable, Grammar, Environment

import haar

try:
    import _ggfe_image_wrap
except ImportError:
    pass
    
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

# Now define the Laws texture energy convolution kernels.
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

def convolve(X, mask):
    "Convolves an image X with a mask"
    return ndi.convolve(X, mask)

def laws(X, i):
    "Computes the 5x5 laws texture energy of an image. i \in \{0,...,24\}"
    return convolve(X, laws_masks[i])

def ptile(X, percentile, struct):
    "Computes the kth percentile filter with a particular structuring element."
    return ndi.percentile_filter(X, percentile, footprint=struct)

def ggm(X, std):
    "Computes the gradient magnitude with Gaussians of a specified width."
    return ndi.gaussian_gradient_magnitude(X, std)

def laplace(X, sigma):
    "Applies a Laplace to an image."
    return ndi.gaussian_laplace(X, sigma)

def sigmoid(I, a, b):
    "Soft maximum with shape parameters a and b."
    return  np.arctan(b * (I + a)) / b

def selem(angle, size, ratio):
    "An elliptical structuring element."
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


def gabor_kernel(angle, size, ratio, period, sigma, Trig):
    "Computes the kernel of the gabor filter, which is just a gaussian multiplied by a sinusoidal."
    [xs, ys] = np.meshgrid(np.arange(-size,size+1), np.arange(-size,size+1))
    shp = xs.shape
    transform = np.dot(np.asarray([[ratio, 0],
                                   [0,     1]]),
                       np.asarray([[np.cos(angle), np.sin(angle)],
                                   [-np.sin(angle), np.cos(angle)]]))
    new_xy = np.dot(transform, np.vstack([xs.ravel(),ys.ravel()]))
    xs = new_xy[0, :]
    ys = new_xy[1, :]
    gaussian = np.exp( -(xs**2. + ys**2.)/(2.*sigma*sigma));
    gabor_trig = gaussian * Trig(xs * np.pi/period);
    gabor_trig = gabor_trig.reshape(shp)
    return gabor_trig

def gabor(X, angle, size, ratio, period, sigma, Trig):
    "Applies a gabor filter of a particular angle, size, aspect ratio, period, sigmal, and sinusoidal."
    gk = gabor_kernel(angle, size, ratio, period, sigma, Trig)
    return convolve(X, gk)

def gabor_sum_squared(X, angle, size, ratio, period, sigma, Trig):
    "Square root of the sum of the modulus squared of two gabor filters."
    gkcos = gabor_kernel(angle, size, ratio, period, sigma, np.cos)
    gksin = gabor_kernel(angle, size, ratio, period, sigma, np.sin)
    return np.sqrt(convolve(X, gkcos) ** 2. + convolve(X, gksin) ** 2.)

def rand1n(n):
    "Generates a random integer between 1 and n (exclusive)."
    return np.random.randint(n)+1

def rand0n(n):
    "Generates a random integer between 0 and n (exclusive)."
    if n == 0:
        return 0
    else:
        return np.random.randint(n)

def vj1x2(w, h):
    "Generates a random 1x2 Viola and Jones kernel."
    return haar.get_kernel_string(haar.Haar2Horizontal(w, h).get_random_rectangles())

def vj2x1(w, h):
    "Generates a random 2x1 Viola and Jones kernel."
    return haar.get_kernel_string(haar.Haar2Vertical(w, h).get_random_rectangles())

def vj1x3(w, h):
    "Generates a random 1x3 Viola and Jones kernel."
    return haar.get_kernel_string(haar.Haar3Horizontal(w, h).get_random_rectangles())

def vj3x1(w, h):
    "Generates a random 3x1 Viola and Jones kernel."
    return haar.get_kernel_string(haar.Haar3Vertical(w, h).get_random_rectangles())

def vj2x2(w, h):
    "Generates a random 2x2 Viola and Jones kernel."
    return haar.get_kernel_string(haar.Haar4(w, h).get_random_rectangles())

def integral_image(I):
    "Computes the integral image of an image."
    _I = np.zeros(I.shape, dtype='f')
    _I[:] = I
    return _ggfe_image_wrap.compute_integral_image_wrap(_I)

def viola_jones(II, s):
    "Given a rectangle specification string s, computes the integral image."
    _II = np.zeros(II.shape, dtype='f')
    _II[:] = II
    return _ggfe_image_wrap.apply_kernel_to_image_wrap(_II, s)

def get_image_grammar(ViolaJonesWidth=24, ViolaJonesHeight=24):
    "Returns the grammar used in the BMVC paper."
    import image_features
    reload(image_features)
    grammar = Grammar("main")

    Feature, Morph, Compound, RandomSE, Gabor, TrigFun = \
       grammar.productions(
           ["Feature", "Morph", "Compound", "RandomSE", "Gabor", "TrigFun"]
           )

    Binary, Unary = grammar.productions(["Binary", "Unary"])
    NLUnary, LUnary = grammar.productions(["NLUnary", "LUnary"])
    NLBinary, LBinary = grammar.productions(["NLBinary", "LBinary"])

    Laws, LawsMask = grammar.productions(["Laws", "LawsMask"])


    erode, dilate, open, close, sigmoid, gabor = \
          functions(["erode", "dilate", "open", "close", "sigmoid", "gabor"],
                     module=image_features)
    mult, normDiff, scaledSub, blend, gabor_sum_squared = \
          functions(["mult", "normDiff", "scaledSub", "blend", "gabor_sum_squared"],
                    module=image_features)

    convolve, laws, laplace, ptile, ggm, selem = \
          functions(["convolve", "laws", "laplace", "ptile", "ggm", "selem"],
                    module=image_features)
    
    normal, random, randint, transpose, dot = \
            functions(["normal", "random", "randint", "transpose", "dot"],
                      module=np.random)

    X, Y, S, W, H = variables(["X", "Y", "S", "W", "H"])

    pi, sin, cos = variables(["pi", "sin", "cos"], expandable=False)

    E5, L5, S5, R5, W5 = variables(["E5", "L5", "S5", "R5", "W5"], expandable=False)

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

    NLUnary[X] = (sigmoid(X, ~normal(), 10 ** (~random() * 2 - 1))
                  | ptile(X, ~(random() * 100), RandomSE[...])
                  | ggm(X, ~normal() * 3)
                  | Morph[X, RandomSE[...]])

    LUnary[X] = (Laws[X]
                 | laplace(X, ~normal() * 3)
                 | Gabor[X])

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


    Gabor[X] = (
               gabor(X,
                     ~(random()) * pi,
                     ~(random() * 30 + 1),
                     10 ** (~(random() * 2 - 1)),
                     ~(random()) * pi,
                     ~(random() * 10 + 2),
                     TrigFun[...])
               |
               gabor_sum_squared(X,
                     ~(random()) * pi,
                     ~(random() * 30 + 1),
                     10 ** (~(random() * 2 - 1)),
                     ~(random()) * pi,
                     ~(random() * 10 + 2),
                     TrigFun[...])
               )
               
    RandomSE[...] = selem((~random()) * pi,
                          ((1 + ~randint(7)) * 2 + 1),
                          (10 ** (~random() * 2 - 1)))

    
    LawsMask[...] = L5 | E5 | S5 | R5 | W5
    
    Laws[X] = convolve(X, dot(transpose(LawsMask[...]), LawsMask[...]))
    
    TrigFun[...] = sin | cos

    # See if the C++ implementation of the Viola and Jones feature set
    # is available. If so, incorporate them into the grammar.
    try:
        import _ggfe_image_wrap
        ViolaJones, KernelString = grammar.productions(["ViolaJones", "KernelString"])
        LUnary += ViolaJones[X]

        viola_jones, integral_image = \
               functions(["viola_jones", "integral_image"])
        vj1x2, vj2x1, vj1x3, vj3x1, vj2x2 = \
               functions(["vj1x2", "vj2x1", "vj1x3", "vj3x1", "vj2x2"],
                         module=image_features)
        
        KernelString[...] = (~vj1x2(ViolaJonesWidth, ViolaJonesHeight)
                             | ~vj2x1(ViolaJonesWidth, ViolaJonesHeight)
                             | ~vj1x3(ViolaJonesWidth, ViolaJonesHeight)
                             | ~vj3x1(ViolaJonesWidth, ViolaJonesHeight)
                             | ~vj2x2(ViolaJonesWidth, ViolaJonesHeight))
        
        ViolaJones[X] = viola_jones(integral_image(X), KernelString[...])
    except ImportError:
        pass
    
    return grammar

def generate_and_evaluate(grammar, IMG):
    """
    Generate a random feature from the grammar passed and evaluate it
    on the image ``IMG``.
    """
    from numpy import pi, sin, cos, dot, transpose
    feature = grammar.Feature(Variable('IMG'))
    print 'Evaluating random feature: ', str(feature)
    return eval(str(feature))
