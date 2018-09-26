class Node:
    def __init__(self, name, layerIndex):
        self.name = name
        self.layerIndex = layerIndex;

class Layer:
    def __init__(self,name):
        self.prevLayer = None
        self.nextLayer = None
        self.nodes = []
        self.name = name
        #print(name)

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
    