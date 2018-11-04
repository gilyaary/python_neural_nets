import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import matplotlib.pylab as plt

env = gym.make('NChain-v0')

def q_learning_keras(env, num_episodes=1000):
    # create the keras model
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 5)))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                #the np.identity(5)[new_s:new_s + 1] just puts a 1 in the position of the state. Example state 3 will become: [0,0,0,1]
                #this is given as an input state. only single 1 in the input vector is the state designator
                #the prediction gives out 2 actions (probability of). We select the inde with a higher value as the selected action
                a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
            new_s, r, done, _ = env.step(a)
            #Now predict the highest action value for the NEXT state
            target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
            
            #it is stupid to execute the multiplication again but the code does it ....
            #this action return an array of arrays. We need jest one array with the 2 predicted action values. Thus we do [0]
            state = np.identity(5)[s:s + 1]
            print('state:', state)
            target_vec = model.predict(state)[0]

            target_vec[a] = target
            #to the fit() function we supply the input (state) and the modified target_vec with the "correct" state action value
            # the reshape is there to change: [q1,q2] to: [[q1,q2]] as Tensorflow wants
            model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / 1000)
    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()
    for i in range(5):
        print("State {} - action {}".format(i, model.predict(np.identity(5)[i:i + 1])))

q_learning_keras(env)
