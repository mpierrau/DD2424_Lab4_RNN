import matplotlib.pyplot as plt
import RNN_model_class
import numpy as np
import read_book

from RNN_model_class import RNN_model
from compute_grads_num import testGrads
from winsound import MessageBeep

bookData , bookChars = read_book.getData('goblet_book.txt')

# Create two equivalent models
mod1 = RNN_model(bookChars,m=100,seed=1337)
mod2 = RNN_model(bookChars,m=100,seed=1337)
mod = RNN_model(bookChars,m=100)
# Fit one using overlap and the other without

mod1.fit(bookData, 2, 25, .1,verbose=False)
MessageBeep()
mod2.fit(bookData, 2, 25, .1, incr=20, overlap=True,verbose=False)
MessageBeep()

# Plot loss
steps1 = range(0,50*len(mod1.loss),50)
steps2 = range(0,50*len(mod2.loss),50)
plt.plot(steps1,mod1.smooth_loss,alpha=0.6,label="Smoothed loss no overlap")
plt.plot(steps2,mod2.smooth_loss,alpha=0.6,label="Smoothed loss overlap 20")
plt.title("Evolution of loss")
plt.xlabel("Step")
plt.ylabel("Loss/Smoothed Loss")
plt.legend()
plt.show()

# Produce synthesized text
X0 = mod2.charToVec['H']
h0 = np.zeros((mod2.m,1))
print(mod2.synthTxt(1000,h0,X0))
print(bookData[:100])

