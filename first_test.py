from importlib import reload

import RNN_model_class
import RNN_funcs
import numpy as np
import compute_grads_num
import read_book

reload(compute_grads_num)
reload(RNN_funcs)
reload(RNN_model_class)
reload(read_book)

from RNN_model_class import RNN_model
from RNN_funcs import softMax , cross_entropy
from compute_grads_num import finite_diff , centered_diff , recurseGrads , relErr , testGrads

bookData , bookChars = read_book.getData('goblet_book.txt')

mod = RNN_model(bookChars,m=100)

mod.fit(bookData, 10, 25, .1)

X , _ = mod.makeOneHot(bookData[700:702])
y = mod.synthTxt(1000,np.zeros((mod.m,1)),X)
print(mod.translateTxt(y))


"""

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

"""