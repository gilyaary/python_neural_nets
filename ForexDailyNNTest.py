# Hello World program in Python
import numpy as np 
from DenseMultiLayerNetworkVer2 import *
#print(L1.weights)
import csv

'''
data = list(csv.reader(open("/home/lilach/MarketStudies/TestConverter_scaled.csv")))
data = np.array(data)
rowSize = np.shape(data)[0]
colSize = np.shape(data)[1]
lastRow = 5000
X = (data[1:lastRow,0:colSize-1])
Y = (data[1:lastRow,colSize-1:colSize])
'''



X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],

    [2,4],
    [4,4],
    [2,2],
    [4,2]
])
Y = np.array([
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0]
])

'sigmoid works better even with layers like 3,3,3,3,1'

LR = 0.5
lm = LayerManager(2, Sigmoid(), LogisticLoss(), LR,32,1)


#LR = 0.01 #LR = 0.0001
#lm = LayerManager(2, ReLU(), SquaredErrLoss(), LR, 2, 1)


#LR = 0.01
#lm = LayerManager(2, ReLU(), SquaredErrLoss(), LR, 20, 1)


for i in range (1, 6000):
    lm.fwd(X)
    lm.back(Y)
lm.predict(Y)
#Note: we must use a slower learning rate for ReLU

