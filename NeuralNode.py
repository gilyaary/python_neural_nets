import numpy as np

class Activation:

    def __init__(self):
        self.name = "Activation"

    'static function'    
    def sigmoid(z):
        return 1 / ( 1 + (np.exp(-z)) )
            
class NeuralNode:

    def __init__(self, inputCount):
        self.inputCount = inputCount
        self.weights = np.full((1,inputCount), 0.1)            
        self.activationFunction = Activation.sigmoid

    def setInputMatrix(self, inputMatrix):
        self.inputMatrix = inputMatrix

    def getZ(self):
        return np.dot(self.weights, np.transpose(self.inputMatrix))

    def getActivation(self):
        z = self.getZ()
        return self.activationFunction(z)


nn = NeuralNode(4)
'''
activation = nn.getActivation()
print("Activation:", activation)
'''
A = np.array([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3],
    [4,4,4,4],
    [5,5,5,5],
    [6,6,6,6],
    [-6,-6,-6,-6]
])
np.set_printoptions(precision=2)
nn.setInputMatrix(A)
z = nn.getZ()
print(z)
a = nn.getActivation()
print(a)

                     
