from Activation import Activation
from ANNBuilder import ANNBuilder
from Loss import Loss
from NeuralNetwork import NeuralNetwork
from PSO import PSO
import numpy as np
import pandas as pd

# Inputs for dataset
dataset_path = None
does_dataset_have_header = None
dataset = None

# Inputs for ANN
layers = []
activationFunc = []
lossFunc = None
optimizer = "pso"
trainSize = None
testSize = None


# Inputs for PSO
numInformants = None
epochs = None
populationSize = None
alpha = None
beta = None
gamma = None
delta = None

# Taking inputs from the user

# Taking inputs for dataset
dateset_path = input("Enter the path of dataset: ")
does_dataset_have_header = input("Does the dataset have header? (y/n): ")

if does_dataset_have_header == 'y':
    dataset = pd.read_csv(dateset_path)
else:
    dataset = pd.read_csv(dateset_path, header=None)

# Asking train and test size
trainSize = int(input("Enter the training size (60): "))
testSize = 100 - trainSize

# Shuffle the dataset to get diverse training and testing sets
dataset = dataset.sample(frac=1, random_state=42)
dataset.reset_index(drop=True, inplace=True)


# Taking trainSize % as train and rest as test
trainSize = int((trainSize / 100) * dataset.shape[0])
testSize = dataset.shape[0] - trainSize

#one hot encoding for multi-class classification
def one_hot_encode(y, unique_classes):
    one_hot_vals = np.zeros((y.shape[0], len(unique_classes)))
    for i, cls in enumerate(unique_classes):
        one_hot_vals[y == cls, i] = 1
    return one_hot_vals

# Splitting dataset into train and test
df_train = dataset.iloc[:trainSize, :]
df_test = dataset.iloc[trainSize:, :]


# check if data is multi-class or binary
isMultiClass = len(df_train.iloc[:, -1].unique()) > 2

# Split the dataset into training and testing here
if isMultiClass:
    unique_classes = df_train.iloc[:, -1].unique()
    #use one hot encoding for multi-class classification for the labels
    X_train = df_train.iloc[:, :-1].values
    y_train = one_hot_encode(df_train.iloc[:, -1].values, unique_classes)
    
    X_test = df_test.iloc[:, :-1].values
    y_test = one_hot_encode(df_test.iloc[:, -1].values, unique_classes)

else:
    #no one hot encoding for binary classification
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values


# Asking for number of hidden layers
num_hidden_layers = int(input("Enter the number of hidden layers: "))

# List of activation functions
activation_ques = '''\n
                    1) Sigmoid\n
                    2) Tanh\n
                    3) ReLU\n
                    4) Softmax\n
                   '''

# Asking number of neurons and activation funciton for output layer
activationFunc.append(int(input("Please enter the activation function for input layer: " + activation_ques)))

# Asking number of neurons and activation funciton for each layer
for i in range(num_hidden_layers):
    layers.append(int(input("Enter the number of neurons in hidden layer " + str(i + 1) + ": ")))
    activationFunc.append(int(input("Please enter the activation function for layer " + str(i+1) + ": " + activation_ques)))

# Update list of activation functions
for i in range(len(activationFunc)):
    if activationFunc[i] == 1:
        activationFunc[i] = Activation.sigmoid
    elif activationFunc[i] == 2:
        activationFunc[i] = Activation.tanh
    elif activationFunc[i] == 3:
        activationFunc[i] = Activation.relu
    elif activationFunc[i] == 4:
        activationFunc[i] = Activation.softmax

# set output layer nodes based on binary or multi-class classification
# for binary classification we will have 1 output node
# for multi-class classification we will have number of output nodes equal to number of classes

if isMultiClass:
    num_classes = len(df_train.iloc[:, -1].unique())
else:
    num_classes = 1

# Set input layer nodes equal to number of features in the dataset
num_cols = X_train.shape[1]

# Set the number of nodes in each layer
num_nodes = [num_cols]
num_nodes.extend(layers)
num_nodes.append(num_classes)

# Multi-class classification :
#       use softmax activation function for output layer
#       use cross entropy loss function
# Binary classification :
#       use sigmoid activation function for output layer
#       use binary cross entropy loss function
if num_classes > 2:
    activation_funcs = activationFunc
    activation_funcs.append(Activation.softmax)
    lossfunction = Loss.cross_entropy
else:
    activation_funcs = activationFunc
    activation_funcs.append(Activation.sigmoid)
    lossfunction = Loss.binary_cross_entropy

# Build the network
network = ANNBuilder.build(num_nodes, activation_funcs)

#Shape of the network
print("=====================================")
print("Shape of the network")
print("=====================================")
for i in range(len(network.layers)):
    print("Layer ", i+1, ": ", network.layers[i].w.shape)
print("=====================================")

# Inputs for PSO
populationSize = int(input("Enter the population size: "))
numInformants = int(input("Enter the number of informants: "))
epochs = int(input("Enter the number of epochs: "))
alpha = float(input("Enter the alpha value: "))
beta = float(input("Enter the beta value: "))
gamma = float(input("Enter the gamma value: "))
delta = float(input("Enter the delta value: "))

# Building the PSO object
pso = PSO(populationSize, (-1,1), network, numInformants)

# Training the network
pso.train(X_train, y_train, lossfunction, epochs, alpha, beta, gamma, delta)

# Evaluate on train set
print("performance on train set")
accuracy, loss = network.evaluate(X_train, y_train, lossfunction)

print("Train Accuracy: ", round(accuracy*100, 4), "%")
print("Train Loss: ", loss)

print("performance on test set")
accuracy, loss = network.evaluate(X_test, y_test, lossfunction)
print("Test Accuracy: ", round(accuracy*100, 4), "%")
print("Test Loss: ", loss)