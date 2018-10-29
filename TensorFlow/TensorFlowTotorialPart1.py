import tensorflow as tf
from tensorflow import keras
import numpy as np

c0 = tf.constant(5.0, dtype=tf.float32)
#print('c0[scalar]:', c0)
c1 = tf.constant([5.0, 6, 7], dtype=tf.float32)
#print('c1[1D Array]:', c1)
c2 = tf.constant([[5.0, 6, 7],[5.0, 6, 7],[5.0, 6, 7],[5.0, 6, 7]], dtype=tf.float32)
#print('c2[2D Matrix]:', c2)
c3 = tf.constant([[[5.0, 6, 7],[5.0, 6, 7],[5.0, 6, 7],[5.0, 6, 7]],[[5.0, 6, 7],[5.0, 6, 7],[5.0, 6, 7],[5.0, 6, 7]],[[5.0, 6, 7],[5.0, 6, 7],[5.0, 6, 7],[5.0, 6, 7]]], dtype=tf.float32)
#print('c3[3D Cube]:', c3)

result1 = tf.multiply(c3,c3)
#print('result1:', result1)
result2 = tf.multiply(result1, result1)
#print('result2:', result2)

session = tf.Session()
result1 = session.run({'result1':result1, 'result2': result2})
#print(result1)
#print(result2)

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1 #Note the plus sign is overloaded and this is actualy a FUTURE. The future is a promissed method call.
out2 = vec + 2 #Note the plus sign is overloaded and this is actualy a FUTURE. The future is a promissed method call.print(session.run(vec))
#print(session.run(vec))
#print(session.run((out1, out2)))


#Example of a placeholder for params supplied later
x = tf.placeholder(dtype=tf.float32)
y = x * x
y2 = x + x
result = session.run(y, feed_dict={x:[2,3,4,5]})
print(result)
result = session.run(y2, feed_dict={x:[3,4,5,6]})
print(result)


