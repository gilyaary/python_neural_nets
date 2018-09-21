import numpy as np

def sigmoid (z):
    return 1/(1+np.exp(-z))

def nn(A, y, test):
    'Each node is a row and each column is a weight for another input'
    W1 = np.array([
        [0.1,0.1], 
        [0.2,0.2]  
    ])
    b1 = np.array([0.1,0.1])

    'Each node is a row and each column is a weight for another input/activation'
    W2 = np.array([
        [0.1,0.1]
    ])
    b2 = np.array([0.1])

    for i in range(0,1):
        _nn(A, y, W1, b1, W2, b2)

    
def _nn(A, y, W1, b1, W2, b2):
    'Step 1: calculate the outputs (a) of each neuron in each layer'
    Z1 = np.dot(A, np.transpose(W1)) + b1
    A1 = sigmoid(Z1)
    'print(A1)'

    Z2 = np.dot(A1, np.transpose(W2)) + b2
    A2 = sigmoid(Z2)
    'print(A2)'

    'y*np.log(A2) + (1-y) * np.log(1-A2)'
    loss = y*np.log(np.transpose(A2)) + (1-y) * np.log(1-np.transpose(A2))
     'we are not interested in the loss only the derivative of the loss'
    'print (loss)'

    'now we can calculate all 3 gradients'
    da2 = (-np.transpose(A2) - loss) / ((1-np.transpose(A2)) * np.transpose(A2))
    print (da2)

    expz2 = np.exp(-Z2)
    print (expz2)
    dz2 =  expz2 / ((1+expz2) * (1+expz2))
    print (dz2)
    dw2 = W2 * dz2 * da2
    print (dw2)
    









'Test the algorithm'
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

test = np.array([
    [3,20],
    [1,12],
    [13,3],
    [12,-3],
    [5,5],
])


y = np.array([0,0,1,1,0,0,1,1,1,0,0])
nn(A, y, test)

