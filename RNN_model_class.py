import numpy as np
from tqdm import tqdm , trange
from importlib import reload
import read_book
reload(read_book)
from RNN_funcs import softMax , custom_choice , cross_entropy
from math import ceil

class RNN_model:

    def __init__(self, char_list, m = 100, eta = .1, seed=None):

        self.chars = char_list
        self.d = len(char_list)
        
        self.m = m
        self.k = self.d

        self.eta = eta

        self.mu = 0
        self.sig = .01

        self.pars = {}
        self.grads = {}
        self.mPars = {   "V" : np.zeros((self.k , self.m)),
                        "W" : np.zeros((self.m , self.m)),
                        "U" : np.zeros((self.m , self.k)),
                        "b" : np.zeros((self.m , 1)),
                        "c" : np.zeros((self.k , 1)) }
        self.loss = []
        self.quotes = []

        self.seed = seed
        np.random.seed(self.seed)

        self.makeDicts()
        self.initPars()


    def forwardProp(self,X,h0):
        n = X.shape[1]
        ht = h0

        A = np.zeros((self.m,n))
        H = np.zeros((self.m,n))
        P = np.zeros((self.k,n))

        for t in range(n):
            ht , at , pt = self.forwardPass(ht,X[:,[t]])
            #print("Shape of  ht: ", np.shape(ht))
            #print("Shape of at: ", np.shape(at))
            #print("Shape of pt: ", np.shape(pt))
            A[:,[t]] = at
            H[:,[t]] = ht
            P[:,[t]] = pt

        return A , H , P


    def forwardPass(self, hPrev, xt):
        """ h0 and m0 have dim m x 1 """ 
        
        at = self.pars["W"] @ hPrev + self.pars["U"] @ xt + self.pars["b"]
        ht = np.tanh(at)
        ot = self.pars["V"] @ ht + self.pars["c"]
        pt = softMax(ot)

        return ht , at , pt


    def computeLoss(self,P,Y,avg=False):
        loss = cross_entropy(Y,P,avg=avg)
        return loss


    def computeGradients(self,X,Y,A,H,P):
        # The H that is fed in has h1 (t=1) in first column
        n = X.shape[1]

        dLdO = -(Y-P).T
        
        dLdH = np.zeros((X.shape[1],self.m))   
        dLdA = np.zeros(A.shape)
        
        dLdH[-1] = dLdO[-1] @ self.pars["V"]
        dLdA[:,-1] = dLdH[-1] * (1-np.tanh(A[:,-1])**2)

        for t in range(n-2,-1,-1):
            dLdH[t] = dLdO[t] @ self.pars["V"] + dLdA[:,t+1] @ self.pars["W"]
            dLdA[:,t] = dLdH[t] * (1 - np.tanh(A[:,t])**2)
        
        tmpH = np.zeros_like(H)
        tmpH[:,1:] = H[:,:-1]       # the first column of this tmpH contains h_0 and the last has h_(tau-1)

        dLdU = np.dot(dLdA, X.T)
        dLdV = np.dot(dLdO.T, H.T)
        dLdW = np.dot(dLdA, tmpH.T)
        
        dLdb = np.sum(dLdA, axis=1, keepdims=True)
        dLdc = np.sum(dLdO.T,axis=1, keepdims=True)

        self.grads["U"] = dLdU
        self.grads["V"] = dLdV
        self.grads["W"] = dLdW
        self.grads["b"] = dLdb
        self.grads["c"] = dLdc

        self.clipGrads()

    def updatePars(self,eta,eps=np.finfo(float).eps):
        for k in self.mPars.keys():
            self.mPars[k] += (self.grads[k]**2)
            self.pars[k] -= eta / ( np.sqrt(self.mPars[k] + eps) ) * self.grads[k]
        
    def clipGrads(self,lim=5):
        for k in self.grads.keys():
            self.grads[k] = np.maximum( np.minimum( self.grads[k] , lim ), (-lim) )

    def initPars(self):
        self.pars["b"] = np.zeros((self.m,1))
        self.pars["c"] = np.zeros((self.k,1))

        self.pars["U"] = np.random.normal(self.mu,self.sig,(self.m,self.k))
        self.pars["W"] = np.random.normal(self.mu,self.sig,(self.m,self.m))
        self.pars["V"] = np.random.normal(self.mu,self.sig,(self.k,self.m))


    def synthTxt(self,n,h0,x0):
        
        H = h0
        X = x0
        seq = ''
        for _ in range(n):
            H , _ , P = self.forwardPass(H , X)
            xIdx = np.random.choice(self.k,p=P.flatten())
            X = self.indToVec[xIdx]
            char = self.indToChar[xIdx]
            seq += char

        return seq

    def fit(self,bookData,nEpochs,seqLen,eta,resume=False):
        rec1 = 50
        rec2 = 5000
        rec3 = 5000

        self.data = bookData
        n = len(self.data)

        if not(resume):
            self.it = 0
            self.smooth_loss = 0

        for _ in trange(nEpochs):
            e = 0
            hprev = np.zeros((self.m,1))

            while(e < n - seqLen - 1):
                X , Y = self.makeOneHot(self.data[e:e + seqLen])
                
                e += seqLen

                A , H , P = self.forwardProp(X,hprev)
                hprev = H[:,[-1]]

                self.computeGradients(X, Y, A, H, P)

                loss_t = self.computeLoss(P , Y)

                self.updatePars(eta=eta)

                if self.it % rec1 == 0:
                    # save smooth loss
                    self.smooth_loss = loss_t if (self.smooth_loss == 0) else np.average([self.smooth_loss,loss_t],weights=[.999,.001])
                    self.loss.append(self.smooth_loss)

                if self.it % rec2 == 0:
                    print("Smooth loss at iteration {0} : {1}".format(self.it,self.smooth_loss))
                if self.it % rec3 == 0:
                    txt = self.synthTxt(200,hprev,X[:,[0]])
                    self.quotes.append(txt)
                    print("Synthesized text at iteration {0}:".format(self.it))
                    print(txt)
            
                self.it += 1


    def makeOneHot(self,data):
        Xchars = data[: -1]
        Ychars = data[1 : ]
        n = len(Xchars)
        Xmat = np.zeros((self.d,n))
        Ymat = np.zeros((self.d,n))

        for i in range(n):
            Xmat[:,[i]] = self.charToVec[Xchars[i]]
            Ymat[:,[i]] = self.charToVec[Ychars[i]]

        return Xmat , Ymat

        
    def makeDicts(self):

        uniqueChars = self.chars
        uc_len = len(uniqueChars)
        idx = range(len(uniqueChars))

        newVec = np.identity(uc_len)
        
        indToChar = dict(zip(idx,uniqueChars))
        indToVec = {k: newVec[:,[k]] for k in indToChar.keys()}
        charToInd = {v: k for k, v in indToChar.items()}
        charToVec = {k: indToVec[charToInd[k]] for k in charToInd.keys()}

        self.indToVec = indToVec
        self.indToChar = indToChar
        self.charToInd = charToInd
        self.charToVec = charToVec

