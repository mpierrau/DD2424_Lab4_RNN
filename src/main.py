import matplotlib.pyplot as plt
import numpy as np
from read_book import getData
from RNN_model_class import RNN_model
from compute_grads_num import testGrads

# Load data
bookData , bookChars = getData('data/goblet_book.txt')

# Create network
RNN = RNN_model(bookChars, m = 100)

# Check gradients
X , Y = RNN.makeOneHot(bookData[:26])
errs , anNet , numNet = testGrads(RNN, X, Y)
errs

# Try to overfit model on small dataset
RNN.fit(bookData[:100], 5000, 25, .1)

# Produce synthesized text
X0 = RNN.charToVec['\t']
h0 = np.zeros((RNN.m, 1))
print(RNN.synthTxt(100, h0, X0))
print(bookData[:100])

# Run long-ish training stint (~120 000 steps)
RNN.fit(inputData=bookData, 
        nEpochs = 3, 
        seqLen = 25, 
        eta = .1,
        recLossEvery = 50,
        printLossEvery = 5000,
        printQuoteEvery = 5000)

# RNN.fit(bookData, 4, 25, .1, resume=True, verbose=False)

# Plot loss
steps = range(0, 50*len(RNN.loss), 50)
plt.plot(steps, RNN.loss, alpha = 0.3, label = "Loss")
plt.plot(steps, RNN.smoothLoss, alpha = 0.6, label = "Smoothed loss alpha=0.99")
plt.title("Evolution of loss")
plt.xlabel("Step")
plt.ylabel("Loss/Smoothed Loss")
plt.legend()
plt.show()

# Produce synthesized text of length 1000
print(RNN.synthTxt(1000))


