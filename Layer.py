
#Put all in a package called NeuralNet
import numpy as np

class Node:
    def __init__(self, name, layerIndex):
        self.name = name
        self.layerIndex = layerIndex;
        self.ins = []
        self.outs = []
        #We have weights for inputs. One for each in edge 
        self.weights = []
        self.bias = 0.1
        #the activation of the neuralNode
        self.outValue = []
        #we use the name slopes to desribe a gradient. One slope per input
        self.slopes = []
        self.activationFunction = SigmoidActivation()

    def initializeWeights(self):
        inputCount = len(self.ins)
        self.weights = np.random.rand(inputCount)
        self.bias = np.random.rand()

    def fwd(self):
        print(self.name, 'Forward')
        if self.layerIndex > 0:
            inputCount = len(self.ins)
            inputMatrix = np.zeros(( inputCount, len(self.ins[0].outValue) ))
            i = 0
            for input in self.ins:
                #print(input.outValue)
                inputMatrix[i] = input.outValue
                i = i+1
            #print(inputMatrix)
            z = np.dot(inputMatrix.T, self.weights)
            #print(z)
            z = z + self.bias
            #print(z)
            self.outValue = self.activationFunction.apply(z)
            #print(self.outValue)
            
    def back(self):
        print(self.name, 'BackProp')
        
class SigmoidActivation:
    def apply(self, z):
        return 1/(1+np.exp(-z))

class ThresholdActivation:
    def apply(self,z):
        if z > 0:
            return 1
        else:
            return 0

class ReLuActivation:
    def apply(self, z):
        if z > 0:
            return z
        else:
            return 0
    
class Edge:
    def __init__(self, leftNode, rightNode):
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.weight = 0
        
class Layer:
    def __init__(self,name):
        self.prevLayer = None
        self.nextLayer = None
        self.nodes = []
        self.name = name
        #print(name)

    def initializeWeights(self):
        for node in self.nodes:
            node.initializeWeights()
        if(self.nextLayer):
            self.nextLayer.initializeWeights()
    
    def fwd(self):
        for node in self.nodes:
            node.fwd()
        if(self.nextLayer):
            self.nextLayer.fwd()

    def back(self):
        for node in self.nodes:
            node.back()
        if(self.prevLayer):
            self.prevLayer.back()

class LayerBuilder:
    def __init__(self):
        self.layers = []
        self.nodeMap = {}
    def build(self, lines):
        
        for line in lines:
            #print(line)
            segments = line.split('->')
            length = len(segments)
            if length == 2:
                node1Name = segments[0]
                node2Name = segments[1]
                node1 = None
                node2 = None
                #print('Node1: ', segments[0], 'Node2: ', segments[1])

                if not node1Name in self.nodeMap:
                    #this should happen only for the first node (Layer0)
                    node1 = Node(node1Name, 0)
                    self.nodeMap[node1Name] = node1
                    layer = None
                    if len(self.layers) > 0:
                        layer = self.layers[0]
                    else:
                        layer = Layer(0)
                        self.layers.append(layer)
                    layer.nodes.append(node1)
                else:
                    node1 = self.nodeMap[node1Name]

                if not node2Name in self.nodeMap:
                    #this should happen only for the first node (Layer0)
                    node2 = Node(node2Name, node1.layerIndex+1)
                    self.nodeMap[node2Name] = node2
                    layer = None
                    if len(self.layers) > node1.layerIndex+1:
                        layer = self.layers[node1.layerIndex+1]              
                    else:
                        layer = Layer(node1.layerIndex+1)
                        self.layers.append(layer)
                        layer.prevLayer = self.layers[node1.layerIndex]
                        self.layers[node1.layerIndex].nextLayer = layer
                    layer.nodes.append(node2)
                else:
                    node2 = self.nodeMap[node2Name]

                #Now we know that this relation is new so we add it to both nodes
                node1.outs.append(node2)
                node2.ins.append(node1)

def loss(actualLabel, predictedLabel):
    return actualLabel * np.log(predictedLabel) + (1-actualLabel) * (np.log(1-predictedLabel))  


l_1 = Layer('L1')
lb = LayerBuilder()
lines=(
    'Input1->Node_1_1',
    'Input2->Node_1_1',
    'Input3->Node_1_1',
    'Input1->Node_1_2',
    'Input2->Node_1_2',
    'Input3->Node_1_2',
    'Node_1_1->Node_2_1',
    'Node_1_2->Node_2_1'
    )
lb.build(lines)
#print(lb.layers)
#print(lb.nodeMap)
'''
for layer in lb.layers:
    nextLayerName = ''
    prevLayerName = ''
    if layer.prevLayer:
        prevLayerName = layer.prevLayer.name
    if layer.nextLayer:
        nextLayerName = layer.nextLayer.name
        
    print('Layer: ', layer.name, 'Prev:', prevLayerName, 'Next:', nextLayerName)
    for node in layer.nodes:
        print('+', node.name)
    print('-----------')
    

'''
'''
The way this code works is:
    When doing push forward we call layers[0].fwd()
    When doing back propagation we call layers[N-1].back()
    These calls will bring about a chain of subsequent calls:
    Each layer:
       1. calls all nodes in that layer to either calculate the activation (fwd) or the gradient (back)
       2. Calls the next layer (fwd) or previous layer (back)
    
'''
lastLayerIndex = len(lb.layers)-1
#each of the nodes below is actualy an INPUT. It has one value for each example
lb.layers[0].nodes[0].outValue = np.array([1,1,1,1,1,1,1,1,1,-2])
lb.layers[0].nodes[1].outValue = np.array([2,2,2,2,2,2,2,2,2,3])
lb.layers[0].nodes[2].outValue = np.array([3,3,3,3,3,3,3,3,3,-4])
desiredLabel = np.array([1,1,1,1,1,1,1,1,1,0])

lb.layers[0].initializeWeights()


#For each epoch
lb.layers[0].fwd()
output = lb.layers[lastLayerIndex].nodes[0].outValue
loss = loss(desiredLabel, output)
print('last hidden OutValue', output)
print('Loss', loss)
lb.layers[lastLayerIndex].back()



'''
Derivatives:

'''
