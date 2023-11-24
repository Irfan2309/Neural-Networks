from NeuralNetwork import NeuralNetwork
from Layers import Layer
# Class for building the network
class ANNBuilder:

    # Function for building the network
    def build(num_nodes, activation_funcs):
        print("=====================================")
        print("Building the Network")
        print("=====================================")
        # Create a new network
        network = NeuralNetwork()
        # For loop for adding layer to the network
        for i in range(1,len(num_nodes)):
            print("Adding Layer: ", i)
            print("Number of Nodes: ", num_nodes[i])
            print("Number of Input Nodes: ", num_nodes[i-1])
            # Add the layer to the network
            network.append(Layer(num_nodes[i-1], num_nodes[i], activation_funcs[i]))
        print("=====================================")
        return network
