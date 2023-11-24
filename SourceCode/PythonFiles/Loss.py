import numpy as np
# Class for Loss functions
class Loss:

    #Mean Absolute Error
    @staticmethod
    def mae_num(y_true, y_predicted):
        return np.mean(np.abs(y_true - y_predicted))
    
    #Mean Squared Error
    @staticmethod
    def mse_num(y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)
    
    #Binary Cross Entropy
    @staticmethod
    def binary_cross_entropy(y_true, y_predicted):
        epsilon = 1e-15
        #We will take max of i or epsilon so that we don't get log(i) as - infinity as log(0) is -inf
        y_pred_new = [max(i, epsilon) for i in y_predicted]
        #Then we will take min of i and 1 - epsilon so that we dont get log(1 - y_pred) as -inf
        y_pred_new = [min(i, 1 - epsilon) for i in y_pred_new]
        #Converting it into numpy array
        y_pred_new = np.array(y_pred_new)
        return -np.mean(y_true * np.log(y_pred_new) + (1 - y_true) * np.log(1 - y_pred_new))
    
    # Categorical Cross Entropy
    @staticmethod
    def cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping predictions for numerical stability
        if y_true.ndim == 1:  # Handling binary classification
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif y_true.ndim == 2:  # Handling multi-class classification
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    # Hinge Loss
    @staticmethod
    def hinge(y_true, y_pred):
        return np.mean(np.maximum(0, 1 - y_true * y_pred))