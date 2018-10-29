import tensorflow as tf
from tensorflow import keras
import numpy as np
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
'''
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def parse_state(current_state):
    row = 0
    col = 0
    if current_state[0] == 1:
        row = 0
    if current_state[1] == 1:
        row = 1
    if current_state[2] == 1:
        row = 2
    if current_state[3] == 1:
        col = 0
    if current_state[4] == 1:
        col = 1
    if current_state[5] == 1:
        col = 2
    return row, col

def encode_state(row, col):
    current_state = np.zeros(shape=[10], dtype=int)
    current_state[row] = 1
    current_state[3 + col] = 1
    return current_state
    
def get_next_state(current_state, action):
    row, col = parse_state(current_state)
    if action == UP and row > 0:
        row -= 1
    if action == DOWN and row < 2:
        row += 1
    if action == RIGHT and col < 2:
        col += 1
    if action == LEFT and col > 0:
        col -= 1
    return encode_state(row, col)
       
def get_reward(state):    
    if state[2] == 1 and state[5] == 1:
        return 1
    else:
        return -0.1

def set_action(state, action):
    for i in range(0,4):
        state[6+i] = 0
    state[ 6 + action] = 1
    return state

#print(set_action([1,1,1,0,0,0,1,1,1,1], UP))
#print(set_action([1,1,1,0,0,0,1,1,1,1], RIGHT))
#print(set_action([1,1,1,0,0,0,1,1,1,1], DOWN))
#print(set_action([1,1,1,0,0,0,1,1,1,1], LEFT))

'''
print(parse_state([0,1,0,  0,1,0]))
print(get_next_state([0,1,0,  0,1,0], UP))
print(get_next_state([0,1,0,  0,1,0], DOWN))
print(get_next_state([1,0,0,  0,1,0], UP))
print(get_next_state([0,0,1,  0,1,0], DOWN))

print(get_next_state([0,1,0,  0,1,0], RIGHT))
print(get_next_state([0,1,0,  0,1,0], LEFT))
print(get_next_state([0,1,0,  0,0,1], RIGHT))
print(get_next_state([0,1,0,  1,0,0], LEFT))
'''

x = tf.placeholder(dtype=tf.float32, shape=[1,10])
q_true = tf.placeholder(dtype=tf.int32, shape=[1,1])
discount = 0.90

l1 = tf.layers.Dense(units=4, activation=tf.sigmoid)
q1 = l1(x)
l2 = tf.layers.Dense(units=2, activation=tf.sigmoid)
q2 = l2(q1)
l3 = tf.layers.Dense(units=1, activation=tf.sigmoid)
q3 = l3(q2)
q_pred = q3
loss = tf.losses.mean_squared_error(labels=q_true, predictions=q_pred)
optimizer = tf.train.GradientDescentOptimizer(1)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
#first initialize
sess.run(init)
for epoch in range(50):
    #now run 100 times
    current_state = [1,0,0,1,0,0,0,0,0,0]
    for j in range (100):
        #todo: add some noise to the real Q value, reduce noise as epoch gets higher
        q_pred_value = [0,0,0,0]
        q_pred_value[UP] = sess.run(q_pred, feed_dict={x:[set_action(current_state, UP)]})[0,0]
        q_pred_value[RIGHT] = sess.run(q_pred, feed_dict={x:[set_action(current_state, RIGHT)]})[0,0]
        q_pred_value[DOWN] = sess.run(q_pred, feed_dict={x:[set_action(current_state, DOWN)]})[0,0]
        q_pred_value[LEFT] = sess.run(q_pred, feed_dict={x:[set_action(current_state, LEFT)]})[0,0]
        noise = np.random.randn(1,4)*(1./(epoch+1))
        #print(q_pred_value)
        #print(noise)
        q_pred_value += noise
        current_state_q = np.argmax(q_pred_value, axis=None)
        best_action_ind = np.unravel_index(current_state_q, [4,1])[0]
        
        #print(best_action_ind)
        current_state = set_action(current_state, best_action_ind)
        #print(current_state)
        
        next_state = get_next_state(current_state, best_action_ind)
        reward = get_reward(next_state)
        #print(next_state)
        next_state_q = [0,0,0,0]
        next_state_q[UP] = sess.run(q_pred, feed_dict={x:[set_action(next_state, UP)]})[0,0]
        next_state_q[RIGHT] = sess.run(q_pred, feed_dict={x:[set_action(next_state, RIGHT)]})[0,0]
        next_state_q[DOWN] = sess.run(q_pred, feed_dict={x:[set_action(next_state, DOWN)]})[0,0]
        next_state_q[LEFT] = sess.run(q_pred, feed_dict={x:[set_action(next_state, LEFT)]})[0,0]
        next_state_highest_q = np.argmax(next_state_q, axis=None)

        
        actual_q = reward + discount *  next_state_highest_q
        
        actual_q = [[1]] #calculate this
        _, loss_value = sess.run( (train,loss), feed_dict={x:[current_state], q_true:actual_q})
        #print(loss_value)
        current_state = next_state
        
        if epoch == 49:
            #print(loss_value)
            print(current_state)

        if next_state[2] == 1 and next_state[5] == 1:
            break
