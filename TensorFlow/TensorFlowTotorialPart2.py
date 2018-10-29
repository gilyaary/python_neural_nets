import tensorflow as tf
from tensorflow import keras
import numpy as np


#Let's build a Dense network with 2 hidden layers
#3 -> [6 -> 6] -> 2 
#5 dimenssional input, unknown number of rows, Each row is an example of the training set
x = tf.placeholder(dtype=tf.float32, shape=[None, 3])
layer = tf.layers.Dense(units=6, activation=tf.sigmoid)
y = layer(x)
layer2 = tf.layers.Dense(units=6, activation=tf.sigmoid)
yy = layer2(y)
layer3 = tf.layers.Dense(units=2, activation=tf.sigmoid)
yyy = layer3(yy)
predicted_lables = yyy

init = tf.global_variables_initializer()
session = tf.Session()
#initialize weights
session.run(init)
r1 = session.run(predicted_lables, {x:[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]})
loss = tf.losses.mean_squared_error(labels = [[0,0],[0,1],[1,0],[1,1]], predictions=r1)
r2 = session.run(loss)
print(r2)
