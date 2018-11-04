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

class Game:
    def __init__(self):
        self.reset()
    def reset(self):
        self.state = np.array([0,0,0,0,0,0,0,0,0])
    def play (self, action):
        played = False
        if self.state[action] == 0:
            self.state[action] = 1
            played = True
        else:
            played = False
        done = True
        for cubicle_state in self.state:
            if cubicle_state == 0:
                done = False
                break

        reward = -0.01
        if done == False and played == True:
            #First check if we won. If so return a reward of 10 points
            player_wins = self.check_for_win(1)
            #If not won then play the opponent and check if he won. If he did return -10
            if player_wins == True:
                #print ('Player Wins')
                reward = 10
                done = True
            else:
                self.opponent_play()
                opponent_wins = self.check_for_win(-1)
                if opponent_wins == True:
                    #print ('Opponent Wins')
                    reward = -10
                    done = True
                        
        return self.state, reward, done

    def get_current_state(self):
        return self.state

    def opponent_play(self):
        for i in range (100):
            opponent_action = np.random.randint(0, 9)
            if self.state[opponent_action] == 0:
                #print ('opponent_action', opponent_action)
                self.state[opponent_action] = -1
                break
        return self.state

    def check_for_win(self, value):
        if self.state[0] == value and self.state[1] == value and self.state[2] == value:
            return True
        if self.state[3] == value and self.state[4] == value and self.state[5] == value:
            return True
        if self.state[6] == value and self.state[7] == value and self.state[8] == value:
            return True
        if self.state[0] == value and self.state[3] == value and self.state[6] == value:
            return True
        if self.state[1] == value and self.state[4] == value and self.state[7] == value:
            return True
        if self.state[2] == value and self.state[5] == value and self.state[8] == value:
            return True
        if self.state[0] == value and self.state[4] == value and self.state[8] == value:
            return True
        if self.state[2] == value and self.state[4] == value and self.state[6] == value:
            return True
        return False

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

agent =  DQNAgent(9,9)
game = Game()
number_of_episodes = 10000
total_rewards = 0
for episode_count in range(number_of_episodes):
    game.reset()
    game.opponent_play()
    agent.clear_memory()
    for step_number in range(10):
        current_state = game.get_current_state()
        action_q_values = agent.act(current_state)

        #reduce the exploration and increase exploitation as time goes by
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        selected_action = np.argmax(action_q_values)
        #generate some random action from time to time
        if np.random.random() < agent.epsilon:
            selected_action = np.random.randint(0, 9)
            #print(selected_action)


        #print('selected_action', selected_action)
        next_state, reward, done = game.play(selected_action)
        agent.add_to_memory(current_state, selected_action, reward, next_state, done)
        current_state = next_state
        #calculate expected q and add experience to memory
        #add the experience tuple. When done adjust gradient with the batch of experiences
        
        if done == True:
            #print ('Episode Done')
            #print (current_state)
            break

    if episode_count%100 == 0:
        print('average_reward', total_rewards/(episode_count+1))     

    #Now take the experiences from memory and adjust weights
    #sample = random.sample(agent.memory, 2)
    sample = agent.memory
    
    
    for current_state, selected_action, reward, next_state, done in sample:
        
        #predict for next state
        next_state_actions_values = agent.model.predict(np.array([next_state]), )
        target = reward
        total_rewards += reward
        if not done:
            target = reward + agent.gamma * np.amax(next_state_actions_values[0])

        #predict for current state
        current_state_action_values = agent.model.predict(np.array([current_state]), )
        current_state_action_values[0][selected_action] = target
        agent.model.fit(np.array([current_state]), current_state_action_values, epochs=1, verbose=0)
        #selected_action_value = current_state_action_values[selected_action]
                
        
