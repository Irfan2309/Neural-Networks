import numpy as np

#Class for a Whole new Neural Network
class NeuralNetwork:

    # Constructor for Neural Network
    def __init__(self):
        # Initializing layers as empty list
        self.layers = []
    
    # Function to add a layer to the network
    def append(self, layer):
        # Add a new layer to the network
        self.layers.append(layer)
    
    # Function for checking if the value is above threshold or not
    def threshold_function(self, output, threshold=0.5):
        # Apply a threshold to the node outputs to determine the activated nodes
        return np.where(output > threshold, 1, 0)
    
    # Function to perform forward pass for 1D data
    def forward1D(self, x_inp):  #shape is 1,4
        # Looping through each layer
        for layer in self.layers:
            # Calling the forward function
            x_inp = layer.forward(x_inp)
        # Returning the output
        return x_inp
    
    # Function to perform forward pass
    def forward(self, X):
        output = []
        # Looping through each data point
        for x_inp in iter(X):
            for layer in self.layers:
                # Calling the forward function
                x_inp = layer.forward(x_inp)
            output.append(x_inp)
        # Returning the output
        return np.array(output)
    
    def evaluate(self, x, y, loss_func):
        # Forward pass
        output = self.forward(x)

        # Checking for multi-class classification
        if len(self.layers[-1].b) > 1:
            predicted_output = np.argmax(output, axis=1)
            y_class_indices = np.argmax(y, axis=1)
            correct_output = np.sum(predicted_output == y_class_indices)
        else:
            # If single class classification then flatten the output
            output = output.flatten()
            # Apply threshold function
            predicted_output = self.threshold_function(output)
            # Calculate the number of correct outputs
            correct_output = np.sum(predicted_output == y)

        # Calculate loss
        loss = loss_func(y, output)

        # Calculate accuracy
        accuracy = correct_output / len(y)

        # Return accuracy and loss
        return accuracy, loss

    # Function to get the weights and biases
    def weightBiasList(self):
        # Initialize an empty list
        arr = np.array([])
        # Looping through each layer
        for layer in self.layers:
            # Appending the weights and biases to the list
            arr = np.append(arr, layer.get_weights())
            arr = np.append(arr, layer.get_bias())
        
        # Returning the list
        return arr.flatten()
       
    
    # Function to update the weights and biases
    def update(self, wb_array):
        # Initialize an index
        index = 0
        # Looping through each layer
        for layer in self.layers:
            # Updating the weights and biases
            index = layer.update(wb_array, index)
    
