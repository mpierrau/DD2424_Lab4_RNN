import numpy as np
from tqdm import tqdm , trange
from importlib import reload
import read_book
reload(read_book)
from RNN_funcs import softMax , custom_choice
from read_book import parseBook , makeDicts

class RNN_model:

    def __init__(self, fileName, m = 100, eta = .1, seqLen = 25, seed=None):

        self.data , self.chars , self.d = parseBook(fileName)
        self.indToVec , self.indToChar , self.charToInd , self.charToVec = makeDicts(self.data)
        
        self.m = m
        self.k = self.d

        self.eta = eta
        self.seqLen = seqLen

        self.mu = 0
        self.sig = .01

        self.pars = {"U" : None , "W" : None ,"V" : None , "b" : None , "c" : None}
        self.grads = {"U" : None , "W" : None ,"V" : None , "b" : None , "c" : None}

        self.seed = seed
        np.random.seed(self.seed)

        self.initPars()

    def forwardProp(self,X,h0):
        A = np.zeros((self.m,self.seqLen))
        H = np.zeros((self.m,self.seqLen))
        P = np.zeros((self.k,self.seqLen))

        ht = h0

        for t in range(self.seqLen):
            ht , at , pt = self.forwardPass(ht,X[:,[t]]) 
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

    def computeGrads(self,X,Y,A,H,P):
        # The H that is fed in has h1 (t=1) in first column

        dLdO = -(Y-P).T

        dLdC = np.sum(dLdO,axis=1, keepdims=True)
         
        dLdV = dLdO.T @ H.T # CHECK
        
        dHdA = np.diag(1-np.tanh(A**2))
        
        dLdH = np.zeros((X.shape[1],self.m))   
        dLdA = np.zeros(A.shape)
        dLdW = np.zeros(self.pars["W"].shape)
        
        dLdH[:,-1] = dLdO[:,[-1]] @ self.pars["V"]
        dLdA[:,-1] = dLdH[:,[-1]] @ dHdA[:,[-1]]

        for t in range(X.shape[1]-2,-1,-1):
            dLdH[:,[t]] = dLdO[:,[t]] @ self.pars["V"] + dLdA[:,[t+1]] @ self.pars["W"]
            dLdA[:,[t]] = dLdH[:,[t]] @ dLdA[:,[t]]
        
        tmpH = np.zeros_like(H)
        tmpH[:,1:] = H[:,:-1]

        dldW = dLdA @ H.T

        dLdU = dLdA.T @ X.T        
        dLdB = np.sum(dLdA, axis=1, keepdims=True)


        #dLdW 

        return 




    def initPars(self):
        self.pars["b"] = np.zeros((self.m,1),dtype = int)
        self.pars["c"] = np.zeros((self.k,1),dtype = int)

        self.pars["U"] = np.random.normal(self.mu,self.sig,(self.m,self.k))
        self.pars["W"] = np.random.normal(self.mu,self.sig,(self.m,self.m))
        self.pars["V"] = np.random.normal(self.mu,self.sig,(self.k,self.m))

    def synthTxt(self,n):
        
        Y = np.zeros( (self.d , n) , dtype = int)
        H = np.zeros( (self.m , 1) , dtype = int)

        X = self.charToVec['\t']
        
        for i in range(n):
            H , _ , P = self.forwardPass(H , X)
            xIdx = custom_choice(self.d,P[:,0])
            X = self.indToVec[xIdx]

            Y[:,[i]] = X
        
        return Y

    def translateTxt(self,Y):
        n = Y.shape[1]
        seq = ''
        for i in range(n):
            seq += self.indToChar[np.argmax(Y[:,i])]

        return seq

    def fit(self):
        Xchars = self.data[:self.seqLen]
        Ychars = self.data[1:self.seqLen + 1]

        Xmat = np.zeros((self.d,self.seqLen), dtype = int)
        Ymat = np.zeros((self.d,self.seqLen), dtype = int)

        for i in range(len(Xchars)):
            Xmat[:,[i]] = self.charToVec[Xchars[i]]
            Ymat[:,[i]] = self.charToVec[Ychars[i]]

        # TODO> for some length - do:
        forwardProp(Xmat)
        backwardProp(Xmat)
        computeGradients()
