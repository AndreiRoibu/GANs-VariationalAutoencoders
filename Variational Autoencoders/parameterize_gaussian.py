import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

def softplus(x):
    """ Defines the softplus activation function (a smooth ReLU)
    softplus(a) = log(1 + exp(a))
    """
    return np.log1p(np.exp(x))
    
# We create our 1-leayered neural network, with layers (4,3,2)
# Output needs to be 2 to enable plotting the outputs as a scatter plot
# We ignore the bias, for simplicity

W1 = np.random.randn(4, 3)
W2 = np.random.randn(3, 2*2) # 2 outputs for mean, and 2 for standard deviation

def forward(x, W1, W2):
    """Function defining the foward propagation
    """
    hidden = np.tanh(x.dot(W1))
    output = hidden.dot(W2)
    mean = output[:2]
    stddev = softplus(output[2:])
    return mean, stddev

# We use a random input vector

x = np.random.randn(4)

mean, stddev = forward(x, W1, W2)
print("mean:", mean)
print("stddev:", stddev)

# MVN takes in the covariance, hence the ^2 of of the standard deviation
samples = mvn.rvs(mean = mean, cov = stddev**2, size=10000)

# Confirming data has mean and stddev calculated by the network
plt.scatter(samples[:,0], samples[:,1], alpha=0.5)
plt.show()

