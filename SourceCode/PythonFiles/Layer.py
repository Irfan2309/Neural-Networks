import numpy as np
# Class for each layer
class Layer:

    # Constructor for Layer Class
    def __init__(self, num_inputs, num_nodes, activation_func):
        # Setting the random seed to get same results
        np.random.seed(42)
        # Setting number of inputs for each layer
        self.num_inputs = num_inputs
        # Setting number of nodes for each layer
        self.num_nodes = num_nodes

        # Initializing weights and bias with random values
        self.w = np.random.uniform(-1, 1, (num_inputs, num_nodes))  #4x7
        self.b = np.random.uniform(-1, 1, num_nodes)
        
        # Setting activation function for each layer
        self.activation_func = activation_func

    #FUnciton to get the weighted sum and activation value
    def forward(self, x):
        # Calculating the weighted sum
        self.weighted_sum = np.dot(x, self.w) + self.b
        # Calculating the activation function
        self.act_node_val = self.activation_func(self.weighted_sum)
        # Returning the activated value
        return self.act_node_val
    
    #Function to get the flattened weights
    def get_weights(self):
        return self.w.flatten()
    
    #Function to get the flattened bias
    def get_bias(self):
        return self.b.flatten()
    
    #Function to update the weights and bias
    def update(self, wb_array, index):
        # Getting the number of nodes and inputs
        num_nodes = self.w.shape[1]
        num_inputs = self.w.shape[0]

        # Getting the total number of weights and bias
        total_weights =  num_nodes * num_inputs
        total_bias = num_nodes
        total = num_inputs * num_nodes + total_bias

        # Getting the new weights and bias
        cut = index + total
        new_wb = wb_array[index : cut]

        # Updating the weights and bias
        self.w = new_wb[:total_weights].reshape((num_inputs, num_nodes))
        self.b = new_wb[total_weights:]

        index += total
        return index
