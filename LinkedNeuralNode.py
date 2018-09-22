import numpy as np

class Activation:

    def __init__(self):
        self.name = "Activation"

    'static function'    
    def sigmoid(z):
        return 1 / ( 1 + (np.exp(-z)) )
            
class LinkedNeuralNode:

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

    def getGradient(self):
        '''
        TODO: We need to calculate the gradiant dloss/dw  by following these steps: '
        1: get the gradient from the NEXT neural node/s - it is the edge that connect this node to the next
            When this node is connected to MULTIPLE inputs the sum of their respective derivatives is what we need. 
        2. Calculate the gradient: dloss_dw = da/dw * sum(dloss/dw_nextNode)
        '''
        z = self.getZ()
        expz = np.exp(-z)
        #(m,1)
        nextGradient = [1,1,1,1,1,1,1]
        #(m,1)
        dz = (expz / np.square(1+expz)) * nextGradient
        #print(dz)
        rowCount = np.shape(self.inputMatrix)[0]
        inputColumnsCount = np.shape(self.inputMatrix)[1]
        dzTiled = np.tile(dz,(inputColumnsCount, 1))
        dw = self.inputMatrix * dzTiled.T
        db = dzTiled 
        #Now calculate the average slope in each input to get 1*inputColumnsCount vector
        #the previous stage will use the correct slope by index
        avgDw = dw.mean(0)
        avgDb = np.sum(db) / rowCount
        
        return {'dw':avgDw, 'db':avgDb}

    def adjustWeights(self):
        'internal adjustment no need to expose'
        return 0

nn = LinkedNeuralNode(4)
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
'''
z = nn.getZ()
print(z)
a = nn.getActivation()
print(a)
'''
dw = nn.getGradient()
print(dw)

          
