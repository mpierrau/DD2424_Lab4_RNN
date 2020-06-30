import numpy as np
from tqdm import trange
from RNN_funcs import softMax , cross_entropy
from math import ceil
from datetime import datetime
from winsound import MessageBeep
from read_book import makeDicts

class RNN_model:
    """ Vanilla Recurrent Neural Network model class. 
        Uses AdaGrad optimizer.
        
        charList    = dictionary with every unique character that exists in the dataset mapped to an unique integer.
                    - type : dict
                    
        m           = dimension of hidden state
                    - type : int > 0

        mu          = hyperparameter for parameter initialization
                    - type : float

        sig         = hyperparameter for parameter initialization
                    - type : float

        seed        = random seed for reproducability
                    - type : int
        """

    def __init__(self, charList, m = 100, mu = 0, sig = .01, seed=None):

        self.chars = charList
        self.d = len(charList)
        
        self.m = m
        self.k = self.d     # This is not always the case. Only when indim = outdim.


        self.mu = mu
        self.sig = sig

        self.pars = {}
        self.grads = {}
        self.mPars = {}

        self.loss = []
        self.smoothLoss = []
        self.quotes = []

        self.seed = seed
        np.random.seed(self.seed)

        self.indToVec , self.indToChar , self.charToVec = makeDicts(charList)
        self.initPars(self.mu,self.sig)


    def initPars(self,mu,sig):
        self.pars["b"] = np.zeros((self.m,1))
        self.pars["c"] = np.zeros((self.k,1))

        self.pars["U"] = np.random.normal(mu,sig,(self.m,self.k))
        self.pars["W"] = np.random.normal(mu,sig,(self.m,self.m))
        self.pars["V"] = np.random.normal(mu,sig,(self.k,self.m))

        self.mPars = {k: np.zeros(np.shape(self.pars[k])) for k in self.pars.keys()}


    def forwardProp(self,X,h0):
        n = X.shape[1]
        ht = h0

        A = np.zeros((self.m,n))
        H = np.zeros((self.m,n))
        P = np.zeros((self.k,n))

        for t in range(n):
            ht , at , pt = self.forwardPass(ht,X[:,[t]])
            
            A[:,[t]] = at
            H[:,[t]] = ht
            P[:,[t]] = pt

        return A , H , P


    def forwardPass(self, hPrev, xt):
        """ hPrev is the hidden state from the previous timestep """

        at = self.pars["W"].dot(hPrev) + self.pars["U"].dot(xt) + self.pars["b"]
        ht = np.tanh(at)
        ot = self.pars["V"].dot(ht) + self.pars["c"]
        pt = softMax(ot)

        return ht , at , pt


    def computeGradients(self,X,Y,A,H,P):
        """ The argument H has h1 (t=1) in its first column """

        n = X.shape[1]

        dLdO = -(Y-P).T
        
        dLdH = np.zeros((H.shape[1],H.shape[0]))   
        dLdA = np.zeros(A.shape)
        
        dLdH[-1] = dLdO[-1].dot(self.pars["V"])
        dLdA[:,-1] = dLdH[-1] * (1-np.tanh(A[:,-1].T)**2)

        for t in range(n-2,-1,-1):
            dLdH[t] = dLdO[t].dot(self.pars["V"]) + dLdA[:,[t+1]].T.dot(self.pars["W"])
            dLdA[:,t] = dLdH[t] * (1 - np.tanh(A[:,t].T)**2)
        
        tmpH = np.zeros_like(H)
        tmpH[:,1:] = H[:,:-1]       # The first column of tmpH contains h_0 and the last has h_(tau-1). This allows for matrix multiplication.

        dLdU = np.dot(dLdA, X.T)
        dLdV = np.dot(dLdO.T, H.T)
        dLdW = np.dot(dLdA, tmpH.T)
        
        dLdb = np.sum(dLdA, axis=1, keepdims=True)
        dLdc = np.sum(dLdO.T, axis=1, keepdims=True)

        self.grads["U"] = dLdU
        self.grads["V"] = dLdV
        self.grads["W"] = dLdW
        self.grads["b"] = dLdb
        self.grads["c"] = dLdc

        self.clipGrads()


    def updatePars(self,eta,eps=1e-8):
        """ AdaGrad optimization. eps is to avoid null division. """

        for k in self.mPars.keys():
            self.mPars[k] += (self.grads[k]**2)
            self.pars[k] -= eta / ( np.sqrt(self.mPars[k] + eps) ) * self.grads[k]


    def clipGrads(self,lim=5):
        """ We need to clip the gradients to avoid exploding gradients. 
            The limit 5 is set from the assignment instructions. """

        for k in self.grads.keys():
            self.grads[k] = np.maximum( np.minimum( self.grads[k] , lim ), (-lim) )


    def computeLoss(self,P,Y,avg=False):
        loss = cross_entropy(Y,P,avg=avg)
        return loss


    def fit(self,   inputData, nEpochs, seqLen, eta, 
                    recLossEvery=50, printLossEvery=5000, printQuoteEvery=5000, 
                    incr=None, overlap=False, resume=False, save=True, verbose=True):

        """ Fit model to data.
        
            inputData        = the entire text to be trained on. In this case the entire book. 
                            - type : string 
                        
            nEpochs         = number of epochs to train the network. One epoch = one complete "readthrough" of inputData.
                            - type : int
                        
            seqLen          = length of each character sequence to be used in minibatch. 
                            - type : int
                        
            eta             = learning rate
                            - type : float 

            recLossEvery    = record loss every recLossEvery steps.
                            - type : int 

            printLossEvery  = if verbose == True, print smoothed loss every printLossEvery steps.
                            - type : int

            printQuoteEvery = if verbose == True, print a synthesized text sequence every printQuoteEvery steps.
                            - type : int
                        
            incr            = increment. Value deciding how many characters to jump ahead after finishing the previous sequence.
                            If incr < seqLen nearby sequences will overlap. 
                            If incr > seqLen some data will be skipped. 
                            If incr == seqLen or incr is None data will be read without skip or overlap. 
                            Only applies if overlap == True
                            - type int 
                        
            overlap         = see incr.
                            - type : boolean
            
            resume          = allows the network to resume training from where it previously stopped.
                            - type : boolean

            save            = saves: 
                                network loss and smoothed loss to an npz file with name (ddmmyyHHMM_loss.npz).
                                network parameters to another npz file (ddmmyyHHMM_pars.npz).
                            - type : boolean

            verbose         = if true, print some stuff, if false, print no stuff.
                            - type : boolean
            """
        
        self.eta = eta

        self.data = inputData
        n = len(self.data)
        
        if not(overlap):
            incr = seqLen
        if not(resume):
            self.it = 0

        for _ in trange(nEpochs):
            e = 0
            hprev = np.zeros((self.m,1))

            while(e < n - seqLen - 2):
                X , Y = self.makeOneHot(self.data[e:e + seqLen + 1])

                A , H , P = self.forwardProp(X,hprev)
                hprev = H[:,[-1]]

                self.computeGradients(X, Y, A, H, P)

                loss_t = self.computeLoss(P , Y)

                self.updatePars(eta=eta)


                if self.it % recLossEvery == 0:
                    self.loss.append(loss_t)

                    if len(self.smoothLoss) == 0:    
                        self.smoothLoss.append(loss_t)
                    else:
                        self.smoothLoss.append(np.average([self.smoothLoss[-1],loss_t],weights=[.99,.01]))

                if self.it % printLossEvery == 0:
                    if verbose:
                        print("Smooth loss at iteration {0} : {1}".format(self.it,self.smoothLoss[-1]))
                if self.it % printQuoteEvery == 0:
                    txt = self.synthTxt(200,hprev,X[:,[0]])
                    self.quotes.append(txt)
                    if verbose:
                        print("Synthesized text at iteration {0}:".format(self.it))
                        print(txt)
                
                e += incr
                self.it += 1
        
        if save:
            now = datetime.now()
            filename = "savefiles/" + now.strftime("%d%m%y%H%M%S")
            np.savez(filename + '_loss',loss=self.loss,smoothLoss=self.smoothLoss)
            np.savez(filename + '_pars',U=self.pars["U"],V=self.pars["V"],W=self.pars["W"],b=self.pars["b"],c=self.pars["c"])
            if verbose:
                print("Loss and pars saved to {0}".format(filename))

        MessageBeep()


    def makeOneHot(self,data):
        """ Creates a one hot encoding of input data sequence. 
        
            data    - type : string of len > 1 
        """
        
        Xchars = data[: -1]
        Ychars = data[1 : ]

        n = len(Xchars)
        Xmat = np.zeros((self.d,n))
        Ymat = np.zeros((self.d,n))

        for i in range(n):
            Xmat[:,[i]] = self.charToVec[Xchars[i]]
            Ymat[:,[i]] = self.charToVec[Ychars[i]]

        return Xmat , Ymat

        
    def synthTxt(self,n,h0=None,x0=None):
        """ Synthesizes (generates) text snippet of length n using hidden state h0 and initial character x0.
        
            n   = length of sequence to be generated
                - type : int
                
            h0  = hidden state of previous timestep
                - type : numpy array of dim (self.m , 1) 
                
            x0  = one hot encoding of initial character 
                - type : numpy array of dim (self.k , 1) """
        
        if h0 is None:
            h0 = np.zeros((self.m,1))
        
        if x0 is None:
            x0 = self.charToVec('\n')

        H = h0
        X = x0

        seq = self.indToChar[np.argmax(x0)]
        
        for _ in range(n):
            H , _ , P = self.forwardPass(H , X)
            
            xIdx = np.random.choice(self.k,p=P.flatten())
            
            X = self.indToVec[xIdx]

            seq += self.indToChar[xIdx]

        return seq