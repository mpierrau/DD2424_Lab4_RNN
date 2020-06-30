import numpy as np
import math
from scipy.linalg import fractional_matrix_power
import csv

def cross_entropy(Y,P,oneHotEnc=True,avg=True):
    """ Computes average Cross Entropy between Y and P.
        Returns average cross entropy if avg==True. """
    
    assert(Y.shape == P.shape)

    N = np.shape(Y)[1]
    
    if(oneHotEnc):
        entrFunc = lambda Y,P : np.trace(-np.log(np.dot(Y.T,P)))
    else:
        entrFunc = lambda Y,P : np.trace(-np.dot(Y.T,np.log(P)))    
    
    entrSum = entrFunc(Y,P)

    if avg:
        entrSum /= N

    return entrSum


def softMax(X):
    """ Standard definition of the softmax function """
    S = np.exp(X) / np.sum(np.exp(X), axis=0)
    
    return S