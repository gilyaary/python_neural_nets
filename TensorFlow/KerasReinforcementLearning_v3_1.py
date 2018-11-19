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
from SheepAndWolfGame_1 import *
import time



# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        #self.learning_rate = 0.001
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(self.state_size*2, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    def act(self, current_state):
        #print(current_state)
        #print(current_state.shape)
        predicted_action_q_values = self.model.predict(np.array([current_state]), )
        #print(predicted_action_q_values)
        return predicted_action_q_values[0]
    def add_to_memory(self, current_state, selected_action, reward, next_state, done):
        self.memory.append( (current_state, selected_action, reward, next_state, done) )
    def clear_memory(self):
        self.memory = deque(maxlen=2000)

print('creating Agent')
agent =  DQNAgent(64,8)
game = SheepAndWolfGame()
print('creating Game')
number_of_episodes = 100000
total_rewards = 0
last_10_total_rewards = 0
print('Stating Episodes')
episode_count = 0
last_10_episode_count = 0




def get_action(step_number):
    if step_number == 1:
        return 1
    if step_number == 2:
        return 3
    if step_number == 3:
        return 5
    if step_number == 4:
        return 7

    if step_number == 5:
        return 0
    if step_number == 6:
        return 2
    if step_number == 7:
        return 4
    if step_number == 8:
        return 6

    if step_number == 9:
        return 1
    if step_number == 10:
        return 3
    if step_number == 11:
        return 5
    if step_number == 12:
        return 7

    if step_number == 13:
        return 0
    if step_number == 14:
        return 2
    if step_number == 15:
        return 4
    if step_number == 16:
        return 6

    if step_number == 17:
        return 1
    if step_number == 18:
        return 3
    if step_number == 19:
        return 5
    if step_number == 20:
        return 7

    if step_number == 21:
        return 0
    if step_number == 22:
        return 2
    if step_number == 23:
        return 4
    if step_number == 24:
        return 6
        
number_of_episodes = 10

while episode_count < number_of_episodes:
    episode_count += 1
    game.reset()
    #game.opponent_play()
    agent.clear_memory()
    done = False
    step_number = 0

    last_reward = 0

    while step_number < 24:
        step_number += 1
        current_state = game.get_current_state()
        #print(np.reshape(current_state,(8,8)))
        action_q_values = agent.act(current_state)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        selected_action = get_action(step_number)
        next_state, reward, done, valid_move = game.play(selected_action, False)
        #We MUST Move this line here !!!! Otherwise there is now state/action with any reward
        agent.add_to_memory(current_state, selected_action, reward, next_state, done)
        if done == True:
            last_reward = reward
            print('Reward: ', reward)
            break 
        current_state = next_state
        
    #sample = random.sample(agent.memory, 4)
    sample = agent.memory
    for current_state, selected_action, reward, next_state, done in sample:
        #if reward > 0:
            #print('>>>>>>>>>>>>>>>>', np.reshape(current_state,(8,8)))
        #predict for next state
        next_state_actions_values = agent.model.predict(np.array([next_state]), )
        #print('next_state_actions_values', next_state_actions_values)
        target = last_reward
        #total_rewards += reward
        #target = reward + agent.gamma * np.amax(next_state_actions_values[0])
        
        #predict for current state
        current_state_action_values = agent.model.predict(np.array([current_state]), )
        #print('current_state_action_values[0]', current_state_action_values[0])
        print('current_state_action_values[0][selected_action]', current_state_action_values[0][selected_action])
        current_state_action_values[0][selected_action] = target
        print('current_state_action_values[0][selected_action]', current_state_action_values[0][selected_action])
        print('current_state_action_values[0]', current_state_action_values[0])        
        #print('target', target)
        agent.model.fit(np.array([current_state]), current_state_action_values, epochs=1, verbose=0)
        
                
        
