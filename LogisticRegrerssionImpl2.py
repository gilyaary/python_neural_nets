import numpy as np
'''
Calculations:
z = Xw + b
sigmoid = 1/(1+e^-z)
a = sigmoid(z)
dLoss/dA = d/da y*log(a)+(1-y)(1-log(a) = -y/a + (1-y)/1-a)
dA/dZ = d/dz sigmoid(z) = a (1 - a)
dLoss/dZ = dLoss/dA * dA/dZ = a - y

Steps to solve each iteration:
1. We calculate a (vector of N dimensions)
2. We cacluate dLoss/Dz (vector of N dimensions)
3. We calculate dW = x * dLossDz * learningRate

'''

def sigmoid (z):
    return 1/(1+np.exp(-z))


def logisticRegression(A, y):
    rowsCols = np.shape(A)
    np.size(y)
    rowCount = rowsCols[0]
    colCount = rowsCols[1]

    w = np.random.rand(colCount)
    b = np.random.rand()
    learningRate = 0.1
    
    for apoch in range (0, 100):
        'learningRate *= 0.5'
        b = _logisticRegression(A, y, w, b, learningRate)
    return {'weight': w, 'bias': b}

def _logisticRegression(X, y, w, b, learningRate):
    rowsCols = np.shape(X)
    np.size(y)
    rowCount = rowsCols[0]
    colCount = rowsCols[1]

    'Z = 1*m matrix'
    Z = np.dot(X,w) + b
    A = sigmoid(Z)
    dZ = A - y
    db = np.sum(dZ) / rowCount
    dw = np.dot(X.T, dZ) / rowCount
    w -= dw * learningRate
    b -= db * learningRate    
    
    return b


def predict(x, w, b):
    z = np.dot(x,w)
    z += b
    a = sigmoid(z)
    return a

A = np.array([
    [1,10],
    [1,20],
    [8,1],
    [7,2],
    [2,20],
    [3,11],
    [7,-1],
    [12,-1],
    [14,2],
    [2,15],
    [4,10]
])
y = np.array([0,0,1,1,0,0,1,1,1,0,0])
results = logisticRegression(A, y)
print(np.around(results['weight'], decimals=2))

test = np.array([
    [3,20],
    [1,12],
    [13,3],
    [12,-3],
    [5,5],
])

prediction = predict(A, results['weight'], results['bias'])
print( np.around(prediction, decimals=1) )

prediction2 = predict(test, results['weight'], results['bias'])
print( np.around(prediction2, decimals=1) )
