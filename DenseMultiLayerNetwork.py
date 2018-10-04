# Hello World program in Python
import numpy as np 

class Sigmoid:
    def apply(self,Z):
        return 1/(1+np.exp(-Z))
    
class Layer:
    def __init__(self, nodeCount, inputCount, activation):
        self.nodeCount = nodeCount
        self.inputCount = inputCount
        self.activation = activation
        self.Z = None
        self.A = None
        
        #prevLayer node count is the columns of the weight matrix and this layer 
        #node count is the rows in the weight matrix
        self.W = np.ones((self.nodeCount, self.inputCount)) * 0.1
        self.b = np.ones(self.nodeCount) * 0.2
        
    def fwd(self,X):
        self.Z = np.dot(X, self.W.T) + self.b
        #print('Z',self.Z)
        self.A = self.activation.apply(self.Z)
        #print('A',self.A)
        
    def back(self, G):
        print(G)

class LayerManager:
    def __init__(self, inputCount, activation, *nodeCounts):
        self.layers = []
        for i in range(0, len(nodeCounts)):
            self.layers.append(Layer(nodeCounts[i], inputCount, activation))
            #each layer's input_count = previous_layer_node_count (each node has one output)
            inputCount = nodeCounts[i]
    
    def fwd(self, X):
        for i in range (0, len(self.layers)):
            self.layers[i].fwd(X)
            X = self.layers[i].A
        #now the LAST X is the final output. We send it to a Loss function
        #When backpropagating we can use the derivative of that Loss function as the "Input" to the Last layer
        self.YY = X
        #print(X)
    
    def back(self, Y):
        #TODO: this should really be the Gradient of the Cost function
        G = Y - self.YY
        for i in reversed(range (0, len(self.layers))):
            self.layers[i].back(G)

#print(L1.weights)
X = np.array([
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5]
])
Y = np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1]
])

lm = LayerManager(5,Sigmoid(),3,3,1)
lm.fwd(X)
lm.back(Y)
