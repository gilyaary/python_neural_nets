import numpy as np
import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
import matplotlib.pylab as plt
#env = gym.make('NChain-v0')
import collections
from collections import deque
import random
import time

STATES_COUNT = 16
ACTION_COUNT = 2

class SimpleGameTest:

    def __init__(self):
        self.reset()
        self.state = np.zeros(STATES_COUNT)
        
    
    def reset(self):
        #print ('reset')
        self.state = np.zeros(STATES_COUNT)
        self.state[0] = 0
        self.selected_cell = 0  
     
    def play(self, action):
        if action == 1 and self.selected_cell < 15:
            self.state[self.selected_cell] = 0
            self.selected_cell += 1
        if action == 0 and self.selected_cell > 0:
            self.state[self.selected_cell] = 0
            self.selected_cell -= 1
        self.state[self.selected_cell] = 1
        if self.selected_cell == 15:
            return self.state, 10, True
        else:        
            return self.state, -0.1, False
        
        


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        #self.gamma = 1    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        #self.learning_rate = 0.001
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, self.state_size)))
        model.add(Dense(self.state_size*4, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        
        return model
    def act(self, current_state):
        conv_state = np.reshape(current_state, (1,STATES_COUNT))
        #print(conv_state)
        predicted_action_q_values = self.model.predict(np.array(conv_state), )
        return predicted_action_q_values[0]
    def add_to_memory(self, current_state, selected_action, reward, next_state, done):
        self.memory.append( (current_state, selected_action, reward, next_state, done) )
    def clear_memory(self):
        self.memory = deque(maxlen=2000)

game = SimpleGameTest()
#4 inputs for state and 4 outputs for action
number_of_episodes = 100000
episode_count = 0
agent =  DQNAgent(STATES_COUNT,ACTION_COUNT)
total_rewards = 0
last_100_rewards = 0
current_state = None
while episode_count < number_of_episodes:
    if episode_count % 100 == 0:
        print('100 episodes')
    episode_reward = 0
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    game.reset()
    done = False
    loop_count = 0
    while not done and loop_count < 100:
        loop_count += 1
        current_state = game.state.copy()
        action_q_values = agent.act(current_state)
        selected_action = np.argmax(action_q_values)
        if random.random() < agent.epsilon:
            selected_action = random.randrange(ACTION_COUNT)
        #print('selected_action', selected_action)
        next_state, reward, done = game.play(selected_action)
        next_state = game.state.copy()
        agent.add_to_memory(current_state, selected_action, reward, next_state, done)
        episode_reward += reward
    #end of while
    episode_count += 1
    #sample = agent.memory
    if len(agent.memory) >= 4:
        sample = random.sample(agent.memory, 4)
        
        for c_state, selected_action, reward, n_state, done in reversed(sample):
            #predict for next state
            next_state_actions_values = agent.model.predict(np.reshape(n_state, (1,STATES_COUNT)), )
            #target = reward
            #This just destorys it. Also checkout the values of next_state and make sure it is copied
            #reward + agent.gamma * np.amax(next_state_actions_values[0])
            target = episode_reward
            #predict for current state
            c_state_action_values = agent.model.predict(np.array(np.reshape(c_state, (1,STATES_COUNT))), )
            #print('A c_state_action_values', c_state_action_values)
            c_state_action_values[0][selected_action] = target
            #print('B c_state_action_values', c_state_action_values)
            agent.model.fit(np.array(np.reshape(c_state, (1,STATES_COUNT))), c_state_action_values, epochs=1, verbose=0)
            #print('current_state_action_values', c_state_action_values)
            

    #print('episode_reward', episode_reward)
    total_rewards += episode_reward
    last_100_rewards += episode_reward                    
    if episode_count % 100 == 0:
        print('average_reward', total_rewards/(episode_count))
        print('average_100_reward', last_100_rewards/100)
        for i in range (1,4):
            q_values = agent.act(current_state)
            print (q_values)
        last_100_rewards = 0
    agent.clear_memory()
    
        
    
