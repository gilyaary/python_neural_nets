import tensorflow as tf
from tensorflow import keras
import numpy as np

'''
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
#first initialize
sess.run(init)
for i in range(1000):
    #now run 100 times
  _, loss_value = sess.run((train, loss))
  # print(loss_value)

print(sess.run(y_pred))
'''

x = tf.constant(dtype=tf.float32,value=[
    [1,1,1,1],
    [1,1,1,0],
    [1,1,0,1],
    [0,1,1,1],
    [1,0,1,1],
    [0,0,0,0],
    [0,0,1,1],
    [1,1,0,0],
    [1,0,1,0],
    [0,1,0,1],
])
y_true = [[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]

l1 = tf.layers.Dense(units=4, activation=tf.sigmoid)
y1 = l1(x)
l2 = tf.layers.Dense(units=2, activation=tf.sigmoid)
y2 = l2(y1)
l3 = tf.layers.Dense(units=1, activation=tf.sigmoid)
y3 = l3(y2)
y_pred = y3
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(1)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
#first initialize
sess.run(init)
for i in range(1000):
    #now run 100 times
  _, loss_value = sess.run((train, loss))
  if i > 990:
      print(loss_value)

print(sess.run(y_pred))
