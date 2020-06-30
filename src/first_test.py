import matplotlib.pyplot as plt
import RNN_model_class
import numpy as np
import read_book

from RNN_model_class import RNN_model
from compute_grads_num import testGrads

# Load data
bookData , bookChars = read_book.getData('goblet_book.txt')

# Check gradients
mod = RNN_model(bookChars,m=100)

X , Y = mod.makeOneHot(bookData[:26])

errs , anNet , numNet = testGrads(mod, X, Y)
errs

# Try to overfit model on small dataset
mod.fit(bookData[:100],5000,25,.1)

# Produce synthesized text
X0 = mod.charToVec['V']
h0 = np.zeros((mod.m,1))
print(mod.synthTxt(100,h0,X0))
print(bookData[:100])


# Run long training stint (~310 000 steps)
mod.fit(bookData, 7, 25, .1, verbose=False)
mod.fit(bookData, 3, 25, .1, resume=True, verbose=False)
mod.fit(bookData, 5, 25, .1, resume=True, verbose=False)
mod.fit(bookData, 5, 25, .05, resume=True, verbose=False)
# Adjust smoothed loss
sm_loss = [mod.loss[0]]
alpha = 0.99

for l in mod.loss[1:]:
    sm_loss.append(np.average([sm_loss[-1],l],weights=[alpha,1-alpha]))

# Plot loss
steps = range(0,50*len(mod.loss),50)
plt.plot(steps,mod.loss,alpha=0.3,label="Loss")
plt.plot(steps,mod.smooth_loss,alpha=0.6,label="Smoothed loss alpha=0.999")
plt.plot(steps,sm_loss,alpha=0.6,label="Smoothed loss alpha=0.99")
plt.title("Evolution of loss")
plt.xlabel("Step")
plt.ylabel("Loss/Smoothed Loss")
plt.legend()
plt.show()

# Produce synthesized text of length 1000
X0 = mod.charToVec['H']
h0 = np.zeros((mod.m,1))
print(mod.synthTxt(1000,h0,X0))
print(bookData[:100])

