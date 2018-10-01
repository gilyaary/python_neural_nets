
#Put all in a package called NeuralNet
import numpy as np

class Node:
    def __init__(self, name, layerIndex):
        self.name = name
        self.layerIndex = layerIndex;
        self.ins = []
        self.outs = []
        self.outsWeightIndexes = []
        #We have weights for inputs. One for each in edge 
        self.weights = []
        self.bias = 0.1
        #the activation of the neuralNode
        self.outValue = []
        #we use the name slopes to desribe a gradient. One slope per input
        self.slopes = []
        self.activationFunction = SigmoidActivation()
        self.lastNode = False
        self.actualLabel = []
        self.z = 0
        self.inputMatrix = None

    def initializeWeights(self):
        inputCount = len(self.ins)

        self.weights = np.repeat(0.1,inputCount)
        self.bias = 0.1
        #self.weights = np.random.rand(inputCount)
        #self.bias = np.random.rand()

    def fwd(self):
        #print(self.name, 'Forward')
        if self.layerIndex > 0 and not self.lastNode:
            inputCount = len(self.ins)
            self.inputMatrix = np.zeros(( inputCount, len(self.ins[0].outValue) ))
            i = 0
            for input in self.ins:
                #print(input.outValue)
                self.inputMatrix[i] = input.outValue
                i = i+1
            #print(inputMatrix)
            self.z = np.dot(self.inputMatrix.T, self.weights)
            #print(z)
            self.z = self.z + self.bias
            #print(self.z)
            self.outValue = self.activationFunction.apply(self.z)
        if self.lastNode == True:
                #print(self.name, 'Is Last')
                inputValues = self.ins[0].outValue
                #print('****** Result Input ****** : ', inputValues)
                self.outValue = self.actualLabel * np.log(inputValues) + (1-self.actualLabel) * (np.log(1-inputValues))
                print('Prediction:', inputValues)
                print('Actual:', self.actualLabel)
                print('Loss:', self.outValue)

    def back(self):
        print(self.name, 'BackProp')
        if self.lastNode == True:
            #print("LastNodeBackprop")
            inputValues = self.ins[0].outValue
            slope = (self.actualLabel-inputValues) / (inputValues * (-inputValues+1))
            self.slopes = np.array([slope]).T
            #print("LastNodeBackpropSlopes: ", self.slopes)
        if self.lastNode == False and self.layerIndex > 0:
            #print("Backprop")
            #print('OutCount', len(self.outs))
            nextNodesSlopeTotal = None
            for i in range(len(self.outs)):
                #print ("Index: ", i)
                out = self.outs[i]
                weightIndex = self.outsWeightIndexes[i]
                #print('out.slopes', out.slopes)
                if nextNodesSlopeTotal == None:
                    nextNodesSlopeTotal = out.slopes[:,weightIndex]                    
                else:
                    nextNodesSlopeTotal = nextNodesSlopeTotal + out.slopes[:,weightIndex]
            #print('nextNodesSlopeTotal', nextNodesSlopeTotal)
            expZ = np.exp(-self.z);    
            activationGradient = (expZ/(1+np.square(expZ))) #this is da/dz
            #print('activationGradient:', activationGradient)
            slopeAhead = activationGradient * nextNodesSlopeTotal
            #print('nextNodesSlopeTotal: ', nextNodesSlopeTotal)
            #print('activationGradient: ', activationGradient)
            print('slopeAhead:', slopeAhead)
            slopeAheadTiled = np.tile(slopeAhead, (len(self.weights),1))
            #print('slopeAheadTiled:', slopeAheadTiled)
            dw =  slopeAheadTiled * self.inputMatrix #Multiply by the input X to get da/dw = da/dz * dz/dw
            print('dw', dw)
            db = slopeAhead #get the da/db
            #print('db:', db)
            self.slopes = dw.T
            #print('Slopes: ',self.slopes)
            dwAverage = np.sum(dw.T, axis=0)/len(db)
            dbAverage = np.sum(db)/len(db)
            #print('db:', db)
            #print('dwAverage: ',dwAverage)
            #print('dbAverage: ',dbAverage)
            
            self.weights = self.weights + dwAverage * 0.1
            self.bias = self.bias + dbAverage * 0.1

            '''
            for i in range(0,10):
                #print('dw[i]', dw[:,i])
                self.weights = self.weights - dw[:,i]*0.1
                self.bias = self.bias - db[i]*0.1
            '''
            
            #DEBUG BY UN COMMENTING:
            #print('self.weights : ',self.weights)
            #print('self.bias : ',self.bias)
            

    def setLastNode(self, lastNode):
        self.lastNode = lastNode
        
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
                node1.outsWeightIndexes.append(len(node2.ins)) # this is used to know the weight index of the next node we connect to
                node2.ins.append(node1)
        #End of For Loop
        lastLayer = self.layers[len(self.layers)-1]
        for lastNode in lastLayer.nodes:
            lastNode.setLastNode(True)



np.set_printoptions(precision=2)
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
    'Node_1_2->Node_2_1',
    'Node_2_1->Node_3_1'
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
lb.layers[0].nodes[0].outValue = np.array([1,1,1,1,2,9,9,8,8,9])
lb.layers[0].nodes[1].outValue = np.array([1,2,1,2,1,9,8,9,9,8])
lb.layers[0].nodes[2].outValue = np.array([1,1,2,2,1,8,9,7,9,7])
actualLabel = np.array([1,1,1,1,1,0,0,0,0,0])
lb.layers[len(lb.layers)-1].nodes[0].actualLabel = actualLabel

lb.layers[0].initializeWeights()


#For each epoch
for i in range(1000):
    lb.layers[0].fwd()
    lb.layers[lastLayerIndex].back()



'''
Derivatives:

'''
