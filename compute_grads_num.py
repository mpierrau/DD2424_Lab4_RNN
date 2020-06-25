
import numpy as np
from tqdm import trange , tqdm
import copy

def testGrads(net, X, Y, h=1e-4, eps=1e-7, fast=False):
    testNet = copy.deepcopy(net)
    # TODO : include burnin - for this you need update functions in place

    A , H , P = net.forwardProp(X)
    net.computeGradients(X,Y,A,H,P)
    
    anGrads = net.grads

    numGrads = computeGradsNum(testNet, X, Y, h, fast=fast)

    relErrs = {k : relErr(anGrads[k],numGrads[k], eps=eps) for k in anGrads.keys()}

    return relErrs , net , testNet


def computeGradsNum(net, X, Y, h=1e-4, fast=False):
        """ Uses finite or centered difference approx depending on fast boolean 
            Good practice: let net burn in a few steps before computing grads"""
        
        numGrads = {}

        if fast:
            print("Using finite difference method")

            _ , _ , P = net.forwardProp(X)
            c = net.computeLoss(P,Y)

            approx_func = lambda net, key, idxTup : finite_diff(net,key,idxTup,X,Y,c,h)
        else:
            print("Using centered difference method")

            approx_func = lambda net, key, idxTup : centered_diff(net,key,idxTup,X,Y,h)
        
        for key in tqdm(net.pars):
            numGrads[key] = recurseGrads(net,key,approx_func)

        return numGrads


def recurseGrads(net,key,approx_func):
    nDims = len(np.shape(net.pars[key]))
    grad = np.zeros(np.shape(net.pars[key]))

    def recFunc(idx,dim):
        thisDim = dim
        thisIdx = idx
        if thisDim < nDims:
            thisIdx.append(0)

            for i in trange(np.shape(grad)[thisDim],leave=False):
                thisIdx[thisDim] = i
                newIdx = thisIdx
                recFunc(newIdx, thisDim + 1)
            
            del thisIdx[thisDim]
        else:
            thisIdx = tuple(thisIdx)
            #print("Updated grad[%s] from %f to" % (thisIdx,grad[thisIdx]))
            grad[thisIdx] = approx_func(net, key, thisIdx)
            #print(grad[thisIdx])
    
    recFunc([],0)

    return grad



def finite_diff(net,key,idx,X,Y,c,h):
    assert(type(net.pars[key][idx]) is np.float64)

    net.pars[key][idx] += h

    _ , _ , P = net.forwardProp(X)
    c2 = net.computeLoss(P,Y)

    grad_approx = (c2-c) / h

    #reset entries for next pass
    net.pars[key][idx] -= h

    assert(type(grad_approx) is np.float64)

    return grad_approx


def centered_diff(net,key,idx,X,Y,h):
    assert(type(net.pars[key][idx]) is np.float64)
    net.pars[key][idx] -= h       

    _ , _ , P1 = net.forwardProp(X)
    c1 = net.computeLoss(P1,Y)
    
    net.pars[key][idx] += 2*h
    
    _ , _ , P2 = net.forwardProp(X)
    c2 = net.computeLoss(P2,Y)

    grad_approx = (c2 - c1) / (2 * h)

    #reset entries for next pass
    net.pars[key][idx] -= h



    assert(type(grad_approx) is np.float64)
    return grad_approx


def relErr(Wan,Wnum,eps=1e-7):
    # Computes mean relative error between Jacobians Wan and Wnum
    assert(np.shape(Wan) == np.shape(Wnum))
    relErr = np.mean(np.abs(Wan - Wnum)/np.maximum(eps,(np.abs(Wan) + np.abs(Wnum))))

    return relErr
