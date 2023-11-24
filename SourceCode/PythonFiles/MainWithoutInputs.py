from Activation import Activation
from ANNBuilder import ANNBuilder
from Loss import Loss
from NeuralNetwork import NeuralNetwork
from PSO import PSO
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('SourceCode/Dataset/data_banknote_authentication.csv', header=None)

# Shuffle the dataset to get diverse training and testing sets
df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True)


#one hot encoding for multi-class classification
def one_hot_encode(y, unique_classes):
    one_hot_vals = np.zeros((y.shape[0], len(unique_classes)))
    for i, cls in enumerate(unique_classes):
        one_hot_vals[y == cls, i] = 1
    return one_hot_vals

# Split the dataset into training and testing here
df_train = df.iloc[:1000]
df_test = df.iloc[1000:]

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
num_nodes = [num_cols, 7, 4, num_classes]

# Multi-class classification :
#       use softmax activation function for output layer
#       use cross entropy loss function
# Binary classification :
#       use sigmoid activation function for output layer
#       use binary cross entropy loss function
if num_classes > 2:
    activation_funcs = [Activation.sigmoid, Activation.sigmoid, Activation.sigmoid, Activation.softmax]
    lossfunction = Loss.cross_entropy
else:
    activation_funcs = [Activation.sigmoid, Activation.sigmoid, Activation.sigmoid,Activation.sigmoid]
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

# Build the PSO object
pso = PSO(20, (-1,1), network, 3)

# Train the network
pso.train(X_train, y_train, lossfunction , 50, 0.5, 1.4, 1.4, 1.2)

# Evaluate on train set
print("performance on train set")
accuracy, loss = network.evaluate(X_train, y_train, lossfunction)

print("Train Accuracy: ", round(accuracy*100, 4), "%")
print("Train Loss: ", loss)

print("performance on test set")
accuracy, loss = network.evaluate(X_test, y_test, lossfunction)
print("Test Accuracy: ", round(accuracy*100, 4), "%")
print("Test Loss: ", loss)





