import numpy as np
#Class for Activation Functions
class Activation:
    
    #1. Sigmoid: 1 / 1 + e^-x
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    #2. ReLU: max(0,x)
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    #3. Hyperbolic Tangent: tanh
    @staticmethod
    def tanh(x):
        return (np.exp(x) -np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    #4. Softmax: e^x / sum(e^x)
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
