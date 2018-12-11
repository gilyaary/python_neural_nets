import numpy as np
import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras import optimizers
import matplotlib.pylab as plt
#env = gym.make('NChain-v0')
import collections
from collections import deque
import random
import time

#There are 100 states corresponding to 100 places in a 10*10 grid
#some of these cells are blocked, some conain traps, some have bonuses. Getting to a spot in the edge gets you a big reward

STATES_COUNT = 16
ROW_COUNT = 4

class SimpleGameTest:

    def __init__(self):
        self.reset()

    def is_valid_move (self, new_location):
        return True

    def reset(self):
        self.state = np.zeros(STATES_COUNT)
        for i in range(1,1):
            location = self.fill_empty_cell(-1)

        #location = self.fill_empty_cell(1)
        location = 0
        self.state[0] = 1
        
        self.pawn_location = location
        #print('self.state', self.state)
        #print(self.state.reshape(10,10))

    def play(self, action):
        if self.state[(STATES_COUNT-1)] == 1:
            return self.state, 10, True

        #self.display_state()
        
        next_location = -1

        if action == 0:
            next_location = self.pawn_location - ROW_COUNT
        if action == 1 and (self.pawn_location%ROW_COUNT) != (ROW_COUNT-1):
            next_location = self.pawn_location + 1
        if action == 2:
            next_location = (self.pawn_location + ROW_COUNT)
        if action == 3 and (self.pawn_location%ROW_COUNT != 0):
            next_location = self.pawn_location - 1
        if next_location < 0 or (next_location > (STATES_COUNT-1)):
            next_location =  self.pawn_location
        if self.state[next_location] != 0:
            next_location =  self.pawn_location

        if next_location == self.pawn_location:
            return self.state, -0.2, False
        else:
            self.state[self.pawn_location] = 0
            #self.display_state()
            self.pawn_location = next_location
            self.state[self.pawn_location] = 1
            #self.display_state()
            return self.state, -0.1, False

    def fill_empty_cell(self, value):
        #print(self.state.reshape(3,3))
        selected_spot = random.randint(0,STATES_COUNT-1)
        #while self.state[selected_spot] != 0:
        #    selected_spot = random.randint(0,STATES_COUNT-1)
        self.state[selected_spot] = value
        #print(self.state.reshape(3,3))
        return selected_spot

    def display_state(self):
        print(self.state.reshape(3,3))
        
        
    
game = SimpleGameTest()
#game.play(1)
#game.play(2)
#game.play(1)
#game.play(3)
#game.play(0)
#game.play(2)


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.8   # discount rate
        #self.gamma = 1    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0
        self.epsilon_decay = 0.9999
        #self.learning_rate = 0.001
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1,self.state_size)))
        model.add(Dense(self.state_size*8, activation='relu'))
        #model.add(Dense(self.state_size, activation='relu'))
        #model.add(Dense(self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        #model.add(Dense(self.action_size, activation='softmax'))
        #sgd = optimizers.SGD(lr=self.learning_rate, decay=0, momentum=0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        #model.compile(loss='mse',optimizer=sgd)
        #model.add(Activation('softmax'))
        
        return model
    def act(self, current_state):
        conv_state = np.reshape(current_state, (1,self.state_size))
        #print(conv_state)
        predicted_action_q_values = self.model.predict(np.array(conv_state), )
        return predicted_action_q_values[0]
    def add_to_memory(self, current_state, selected_action, reward, next_state, done):
        self.memory.append( (current_state, selected_action, reward, next_state, done) )
    def clear_memory(self):
        self.memory = deque(maxlen=2000)


#4 inputs for state and 4 outputs for action
number_of_episodes = 10000000
episode_count = 0
agent =  DQNAgent(STATES_COUNT,4)
total_rewards = 0
last_N_rewards = 0
current_state = None
while episode_count < number_of_episodes:

    episode_reward = 0
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if agent.epsilon < 0.1:
        agent.epsilon = 0.1

    game.reset()
    done = False
    while not done:
        current_state = game.state.copy()
        action_q_values = agent.act(current_state)
        selected_action = np.argmax(action_q_values)
        rand = random.random()
        if  rand < agent.epsilon:
            selected_action = random.randint(0,3)
        next_state, reward, done = game.play(selected_action)
        next_state = game.state.copy()
        agent.add_to_memory(current_state, selected_action, reward, next_state, done)
        episode_reward += reward
    #end of while


    if episode_count % 10000 == 0 and agent.epsilon<0.9:
        print ('finished another 100 episodes')
        episode_reward = 0
        game.reset()
        done = False
        loop_iterations = 0
        while not done and loop_iterations < 50:
            loop_iterations += 1
            current_state = game.state.copy()
            action_q_values = agent.act(current_state)
            #print(current_state.reshape(4,4))
            #print(action_q_values)
            selected_action = np.argmax(action_q_values)
            next_state, reward, done = game.play(selected_action)
            next_state = game.state.copy()
            episode_reward += reward
        if loop_iterations < 49:
            print('Average Reward:', episode_reward / 10)
        else:
            print('did not complete')
        #end of while
        continue
    #end if
        
    episode_count += 1

    local_memory = deque(maxlen=2000)
    discount = 1
    
    for c_state, selected_action, reward, n_state, done in reversed(agent.memory):
        local_memory.append( (current_state, selected_action, reward, next_state, done) )
        #discount *= agent.gamma         
   
    sample_size = 10
    if len(local_memory) < 10:
        sample = local_memory
    else:
        sample = random.sample(local_memory, (int)(len(local_memory)/2))
    
    for c_state, selected_action, reward, n_state, done in reversed(sample):
        #predict for next state
        next_state_actions_values = agent.model.predict(np.reshape(n_state, (1,agent.state_size)), )
        #target = reward
        #This just destorys it. Also checkout the values of next_state and make sure it is copied
        target = reward + agent.gamma * np.amax(next_state_actions_values[0])
        #target = reward
        #predict for current state
        c_state_action_values = agent.model.predict(np.array(np.reshape(c_state, (1,agent.state_size))), )
        #print('A c_state_action_values', c_state_action_values)
        c_state_action_values[0][selected_action] = target
        #print('B c_state_action_values', c_state_action_values)
        agent.model.fit(np.array(np.reshape(c_state, (1,agent.state_size))), c_state_action_values, epochs=1, verbose=0)
        #print('current_state_action_values', c_state_action_values)
        

    #print('episode_reward', episode_reward)
    total_rewards += episode_reward
    last_N_rewards += episode_reward                    
    if episode_count % 10 == 0:
        #print('average_reward', total_rewards/(episode_count))
        #print('average_100_reward', last_N_rewards/10)
        #print('agent.epsilon', agent.epsilon)
        #print('###########################################')
        last_N_rewards = 0
    agent.clear_memory()
        
    
