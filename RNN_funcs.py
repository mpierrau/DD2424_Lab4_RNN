import numpy as np
import math
from scipy.linalg import fractional_matrix_power
import csv

def cross_entropy(Y,P,oneHotEnc=True,avg=True):
    """ Computes average Cross Entropy between Y and P batchwise """

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


def setEta(epoch,n_s,etaMin, etaMax):
    """ Cyclical learning rate
    
     n_s must be a positive integer
     n_s is typically chosen to be
     k*np.floor(N/n_batch) with k being
     an integer between  2 and 8 and N
     is the total number of training examples
     and n_batch the number of examples in a 
     batch.

     "Normalize" the time so we don't have to
     worry about l """

    t = epoch % (2*n_s)

    if (t <= n_s):
        etat = etaMin*(1-t/n_s) + etaMax*t/n_s
    else:
        etat = etaMax*(2-t/n_s) + etaMin*(t/n_s-1)
    
    return etat

def write_metrics(net,fileName):
    """ Saves loss, cost and accuracy to fileName.csv """

    header = ['Step','Loss','Cost','Accuracy']

    totSteps = 2*net.n_s*net.n_cycles
    
    for key in ["Training","Validation","Test"]:
        steps = range(0,totSteps,net.rec_every)
        
        if len(steps) == 0:
            break

        thisFile = "%s_%s.csv" % (fileName,key)
        
        f = open(thisFile ,"w")
        
        with f:
            writer = csv.writer(f)
            writer.writerow(header)

        f.close()

        f = open(thisFile,"a+")
        
        with f:
            writer = csv.writer(f)
            for i in range(len(net.loss[key])):
                vals = [steps[i] , net.loss[key][i] , net.cost[key][i] , net.accuracy[key][i]]
                writer.writerow(vals)

        f.close()

        print("%s results saved in %s" % (key,thisFile))


def custom_choice(arr,p):
    if np.shape(arr) == ():
        arr = range(arr)

    cp = np.cumsum(p)
    
    val = np.random.rand()

    for i,lim in enumerate(cp):
        if val <= lim:
            idx = i
            break
    
    return arr[idx]
