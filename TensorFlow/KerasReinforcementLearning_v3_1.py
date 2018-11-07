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
        self.learning_rate = 0.01
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.state_size*2, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.state_size*2, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
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
while episode_count < number_of_episodes:
    #print('Stating Episode', episode_count)
    game.reset()
    #game.opponent_play()
    agent.clear_memory()
    done = False
    step_number = 0
    while step_number < 100:
        step_number += 1
        current_state = game.get_current_state()
        action_q_values = agent.act(current_state)
        #print('current_state',current_state)
        #reduce the exploration and increase exploitation as time goes by
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        #agent.epsilon = 1

        #There is an issue where the program gets into very long time stuck in here
        #We should see if theree is a bug saying there are no valid moves for any move
        valid_move = False
        attempt = 0
        while not valid_move:
            attempt += 1
            selected_action = np.argmax(action_q_values)
            #generate some random action from time to time
            if random.random() < agent.epsilon:
                random.seed(random.random())
                selected_action = random.randint(0, 7)
            next_state, reward, done, valid_move = game.play(selected_action)
            if done == True:
                break            
            #if attempt > 100:
            #    print('selected_action', selected_action)

        #calculate expected q and add experience to memory
        #add the experience tuple. When done adjust gradient with the batch of experiences
        agent.add_to_memory(current_state, selected_action, reward, next_state, done)
        current_state = next_state
        if step_number > 95:
            print('valid_move selected_action', selected_action)
            print(np.reshape(game.state,(8,8)))
            valid_move, sheep_current_location, sheep_next_location, sheep_index, direction = game.translate_sheep_action(selected_action)
            print('sheep_current_location', sheep_current_location)
            print('sheep_next_location', sheep_next_location)
            game.sheep_position[sheep_index][0] = sheep_next_location[0]
            game.sheep_position[sheep_index][1] = sheep_next_location[1]
            game.set_state()
            print(np.reshape(game.state,(8,8)))
            
            
        if done == True:
            #print ('Episode Done')
            #print (current_state)
            break
        
    #print('Here')

    if done == True:
        episode_count += 1
        last_10_episode_count += 1
        if episode_count%100 == 0:
            print('average_reward', total_rewards/episode_count)
            print('last 10 average_reward', last_10_total_rewards/100)
            last_10_episode_count = 0
            last_10_total_rewards = 0
    else:
        print ('not done')
        #print(np.reshape(game.state,(8,8)))
        continue


    #print('adjust weights')   
    #Now take the experiences from memory and adjust weights
    sample = random.sample(agent.memory, 4)
    #sample = agent.memory
    
    
    for current_state, selected_action, reward, next_state, done in sample:
        
        #predict for next state
        next_state_actions_values = agent.model.predict(np.array([next_state]), )
        target = reward
        total_rewards += reward
        last_10_total_rewards += reward
        if not done:
            target = reward + agent.gamma * np.amax(next_state_actions_values[0])

        #predict for current state
        current_state_action_values = agent.model.predict(np.array([current_state]), )
        current_state_action_values[0][selected_action] = target
        agent.model.fit(np.array([current_state]), current_state_action_values, epochs=1, verbose=0)
        #selected_action_value = current_state_action_values[selected_action]
                
        
