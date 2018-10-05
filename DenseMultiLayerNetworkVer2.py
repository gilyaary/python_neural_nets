# Hello World program in Python
import numpy as np 

class Sigmoid:
    def apply(self,Z):
        return 1/(1+np.exp(-Z))
    def gradient(self, Z, G):
        exp = np.exp(-Z)
        #print('exp', exp)
        dz = (exp / np.square(1 + exp)) * G
        #print('dz', dz)
        return dz
        

class ReLU:
    def apply(self, X):
        A = (X > 0) * X
        #print('A', A)
        return A
    def gradient(self, X, G):
        dz =  (X > 0) * 1 * G
        #print('dz', dz)
        return dz

class Linear:
    def apply(self, X):
        A = X
        return A
    def gradient(self, X, G):
        #print(X)
        #print(G)
        dz =  X * G 
        return dz        


class SquaredErrLoss:
    def apply(self, Y, YY):
        return np.square(Y-YY)
    def gradient(self, Y, YY):
        return 2*(Y-YY)
    
class LogisticLoss:
    def apply(self, Y, YY):
        return Y * np.log(YY) + (1-Y) * np.log(1-YY)
    def gradient(self, Y, YY):
        return (Y-YY) / (YY*(-YY+1))
        
class LinearLoss:
    def apply(self, Y, YY):
        return Y-YY
    def gradient(self, Y, YY):
        diff = (Y-YY) * 0.0001
        return diff

    
class Layer:
    def __init__(self, nodeCount, inputCount, activation, LR):
        self.nodeCount = nodeCount
        self.inputCount = inputCount
        self.activation = activation
        self.LR = LR
        self.Z = None
        self.A = None
        
        #prevLayer node count is the columns of the weight matrix and this layer 
        #node count is the rows in the weight matrix
        #self.W = np.ones((self.nodeCount, self.inputCount)) * 0.1
        #self.b = np.ones(self.nodeCount) * 0.2
        self.W = np.random.rand(self.nodeCount, self.inputCount) * 0.1
        self.b = np.random.rand(self.nodeCount) * 0.2
        self.X = None
        
    def fwd(self,X):
        self.X = X
        self.Z = np.dot(X, self.W.T) + self.b
        #print('Z',self.Z)
        self.A = self.activation.apply(self.Z)
        #print('A',self.A)
        
    def back(self, G):
        
        dz = self.activation.gradient(self.Z, G)
        #print('self.inputCount', self.inputCount)
        #print('G', G)

        sum_of_all_nodes_gradients = None
        for i in range (0,self.nodeCount):
            nodes_dz = dz[:,i]
            #print('nodes_dz', nodes_dz)
            nodes_dz_tiled = np.tile(nodes_dz, (self.inputCount,1)).T
            #print('nodes_dz_tiled.T Shape', np.shape(nodes_dz_tiled))
            #print('nodes_dz_tiled', nodes_dz_tiled)
            nodes_gradient = self.X * nodes_dz_tiled
            
            
            nodes_average_gradient = sum(nodes_gradient)/np.shape(self.X)[0]
            nodes_average_bias_gradient = sum(nodes_dz)/np.shape(self.X)[0]
            #print('nodes_average_gradient', nodes_average_gradient)
            #print('nodes_average_bias_gradient', nodes_average_bias_gradient)
            self.b[i] = self.b[i] + LR * nodes_average_bias_gradient   
            self.W[i] = self.W[i] + LR * nodes_average_gradient   
            
            #print('nodes_gradient', nodes_gradient)
            if i == 0:
                sum_of_all_nodes_gradients = nodes_gradient
            else:
                sum_of_all_nodes_gradients = sum_of_all_nodes_gradients + nodes_gradient
        #print('sum_of_all_nodes_gradients',sum_of_all_nodes_gradients)
        #print ('Layer Updated W', self.W)
        #print ('Layer Updated bias', self.b)
        return sum_of_all_nodes_gradients
        

class LayerManager:
    def __init__(self, inputCount, activation, loss, LR, *nodeCounts):
        self.activation = activation
        self.loss = loss
        self.LR = LR
        self.layers = []
        for i in range(0, len(nodeCounts)):
            self.layers.append(Layer(nodeCounts[i], inputCount, activation, self.LR))
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
        LOSS = self.loss.apply(Y,self.YY)
        #print(LOSS)
        # looks like multiplying the gradient by a larger factor increases learning rate 
        G = self.loss.gradient(Y,self.YY)
        #print(G)
        for i in reversed(range (0, len(self.layers))):
            G = self.layers[i].back(G)
    
    def predict(self, Y):
        print ('Y', Y)
        print ('YY', self.YY)
        
        
#print(L1.weights)
X = np.array([
    [5,5,5,5,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,3,4],
    [1,2,3,4,4],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,1,3,1,1],
    [1,1,1,1,1]
])
Y = np.array([
    [0],
    [1],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [0],
    [1]
])

'sigmoid works better even with layers like 3,3,3,3,1'
LR = 5
lm = LayerManager(5, Sigmoid(), SquaredErrLoss(), LR, 3,1)
#LR = 0.0001
#lm = LayerManager(5, ReLU(), SquaredErrLoss(), LR, 3, 3, 1)
for i in range (1, 500):
    lm.fwd(X)
    lm.back(Y)
lm.predict(Y)
#Note: we must use a slower learning rate for ReLU
