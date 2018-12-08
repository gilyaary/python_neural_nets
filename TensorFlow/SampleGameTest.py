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

class SimpleGameTest:

    def __init__(self):
        self.reset()
    def is_valid_move (self, sheep_next_xy):
        if sheep_next_xy[0] >= 4:
            return False
        elif sheep_next_xy[1] >= 4:
            return False
        elif sheep_next_xy[1] < 0:
            return False
        for location in self.state:
            if location[0] == sheep_next_xy[0] and location[1] == sheep_next_xy[1]:
                return False
        return True
    def reset(self):
        #we give the location of each sheep as row,col
        self.state = np.array([[0,0],[0,2]]) # we can simulate a wolf with additional pair: , [3,0]
        #print('self.state', self.state)
    def get_sheep_next_xy(self, action):
        sheep_index = (int)(action / 2);
        direction_index = ((int)(action) % 2);
        #print('Sheep: ', sheep_index, 'Direction', direction_index)
        sheep_xy = self.state[sheep_index]
        sheep_next_xy = [sheep_xy[0], sheep_xy[1]]
        sheep_next_xy[0] += 1
        if direction_index == 0:
            sheep_next_xy[1] -= 2
        else:
            sheep_next_xy[1] += 2
        return sheep_index, direction_index, sheep_xy, sheep_next_xy
    def play(self, action):
        sheep_index, direction_index, sheep_xy, sheep_next_xy = self.get_sheep_next_xy(action)
        if self.is_valid_move (sheep_next_xy):
            #print ('valid move')
            self.state[sheep_index] = sheep_next_xy
            #print('self.state', self.state)
            #TODO: Check if we won or lost

            sheep_1_position = self.state[0]
            sheep_2_position = self.state[1]
            if sheep_1_position[0] == 3 and sheep_2_position[0] == 3:
                return self.state, 1, True    
            else:
                return self.state, -0.05, False
        else:
            no_more_moves = True
            for a in range(0,4):
                sheep_index, direction_index, sheep_xy, sheep_next_xy = self.get_sheep_next_xy(a)
                if self.is_valid_move (sheep_next_xy):
                    no_more_moves = False
                    break
            #End of for loop
            if no_more_moves:
                return self.state, -1, True
            else:
                return self.state, -0.05, False
        
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
        conv_state = np.reshape(current_state, (1,4))
        #print(conv_state)
        predicted_action_q_values = self.model.predict(np.array(conv_state), )
        return predicted_action_q_values[0]
    def add_to_memory(self, current_state, selected_action, reward, next_state, done):
        self.memory.append( (current_state, selected_action, reward, next_state, done) )
    def clear_memory(self):
        self.memory = deque(maxlen=2000)


#4 inputs for state and 4 outputs for action
number_of_episodes = 100000
episode_count = 0
agent =  DQNAgent(4,4)
total_rewards = 0
last_100_rewards = 0
current_state = None
while episode_count < number_of_episodes:
    episode_reward = 0
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    game.reset()
    done = False
    while not done:
        current_state = game.state.copy()
        action_q_values = agent.act(current_state)
        selected_action = np.argmax(action_q_values)
        if random.random() < agent.epsilon:
            selected_action = random.randrange(4)
        #print('selected_action', selected_action)
        next_state, reward, done = game.play(selected_action)
        next_state = game.state.copy()
        agent.add_to_memory(current_state, selected_action, reward, next_state, done)
        episode_reward += reward
    #end of while
    episode_count += 1
    #sample = agent.memory
    sample = random.sample(agent.memory, 4)
    
    for c_state, selected_action, reward, n_state, done in reversed(sample):
        #predict for next state
        next_state_actions_values = agent.model.predict(np.reshape(n_state, (1,4)), )
        #target = reward
        #This just destorys it. Also checkout the values of next_state and make sure it is copied
        #reward + agent.gamma * np.amax(next_state_actions_values[0])
        target = episode_reward
        #predict for current state
        c_state_action_values = agent.model.predict(np.array(np.reshape(c_state, (1,4))), )
        #print('A c_state_action_values', c_state_action_values)
        c_state_action_values[0][selected_action] = target
        #print('B c_state_action_values', c_state_action_values)
        agent.model.fit(np.array(np.reshape(c_state, (1,4))), c_state_action_values, epochs=1, verbose=0)
        #print('current_state_action_values', c_state_action_values)
        

    #print('episode_reward', episode_reward)
    total_rewards += episode_reward
    last_100_rewards += episode_reward                    
    if episode_count % 100 == 0:
        print('average_reward', total_rewards/(episode_count))
        print('average_100_reward', last_100_rewards/100)
        last_100_rewards = 0
    agent.clear_memory()
        
    
