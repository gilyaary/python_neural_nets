import numpy as np 
import csv
import tensorflow as tf

test_abs_move = False

data = list(csv.reader(open("/home/lilach/MarketStudies/TestConverter_scaled.csv")))
data = np.array(data)
rowSize = np.shape(data)[0]
colSize = np.shape(data)[1]
print('RowSize:', rowSize, 'ColSize:', colSize)
lastRow = 50001
X = (data[1:lastRow,0:colSize-1])
Y = (data[1:lastRow,colSize-1:colSize])

X_TEST = (data[lastRow:lastRow+1000,0:colSize-1])
Y_TEST = (data[lastRow:lastRow+1000,colSize-1:colSize])


X = X.astype(np.float)
Y = (Y.astype(np.float))
if test_abs_move:
  Y = np.abs(Y)  
#print(X)
#print(Y)
tf_x = tf.constant(dtype=tf.float32,value=X)
y_true = tf.constant(dtype=tf.float32,value=Y)

#Build layers
l1 = tf.layers.Dense(units=20, activation=tf.sigmoid)
y1 = l1(tf_x)
l2 = tf.layers.Dense(units=4, activation=tf.sigmoid)
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
for i in range(1001):
    #now run 100 times
  _, loss_value = sess.run((train, loss))
  if i % 100 == 0:
      print('Iteration: ', i, 'Loss:', loss_value)

y_pred_actual = sess.run( y_pred )
y_pred_flattened = (np.array(y_pred_actual)).flatten()
#print(np.sort(y_pred_flattened))
sum_of_selected_y = 0
count_of_selected_y = 0.00000001
for i in range(0, 50000):
    if y_pred_flattened[i] < 0.010:
        #print(Y[i])
        sum_of_selected_y += Y[i]
        count_of_selected_y += 1
print("Sum", sum_of_selected_y)
print("Count", count_of_selected_y)
print("Avg", sum_of_selected_y/count_of_selected_y)
#print(Y[1:20])

      

