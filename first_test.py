import RNN_model_class
from importlib import reload
reload(RNN_model_class)
from RNN_model_class import RNN_model
import RNN_funcs
from RNN_funcs import softMax
from RNN_funcs import custom_choice
import numpy as np

mod = RNN_model_class.RNN_model('goblet_book.txt')

mod.initPars()

Xchars = mod.data[:mod.seqLen]
Ychars = mod.data[1:mod.seqLen+1]
Xmat = np.zeros((mod.d,mod.seqLen))
Ymat = np.zeros((mod.d,mod.seqLen))
h0 = np.zeros((mod.m,1))

for i in range(len(Xchars)):
    Xmat[:,[i]] = mod.charToVec[Xchars[i]]
    Ymat[:,[i]] = mod.charToVec[Ychars[i]]

A , H , P = mod.forwardProp(Xmat,h0)

X = Xmat
Y = Ymat

dLdO = -(Y.T-P.T).T

dLdC = np.sum(dLdO,axis=1, keepdims=True)

dLdV = dLdO @ H.T

dHdA_tau = np.diag(1-np.tanh(A[:,-1]**2))

dLdH = np.zeros((X.shape[1],mod.m))   
dLdA = np.zeros(A.shape)
dLdW = np.zeros(mod.pars["W"].shape)

dLdH[-1] = dLdO[:,[-1]].T @ mod.pars["V"]
dLdA[:,-1] = dLdH[-1].T*dHdA_tau

for t in range(X.shape[1]-2,-1,-1):
    dLdH[:,[t]] = dLdO[:,[t]] @ mod.pars["V"] + dLdA[:,[t+1]] @ mod.pars["W"]
    dLdA[:,[t]] = dLdH[:,[t]] @ dLdA[:,[t]]

tmpH = np.zeros_like(H)
tmpH[:,1:] = H[:,:-1]

dldW = dLdA @ H.T

dLdU = dLdA.T @ X.T        
dLdB = np.sum(dLdA, axis=1, keepdims=True)

mod.computeGrads()
#


Y = mod.synthTxt(15)
mod.translateTxt(Y)

# n=3
# d=5
# m=4
G = np.array([[1,10,1],[2,20,1],[3,30,1],[4,40,1],[5,50,1]])
H = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])

G[:,[0]]@H[:,[0]].T
G[:,[1]]@H[:,[1]].T
G[:,[2]]@H[:,[2]].T

