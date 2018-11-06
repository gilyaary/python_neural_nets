import numpy as np
import random

LEFT = 1
RIGHT = 2


class SheepAndWolfGame:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = np.array(np.zeros([64]))
        self.sheep_position = np.array([[0,0],[0,0],[0,0],[0,0]])
        self.wolf_position = np.array([0,0])
        for i in range(0,4):
            self.sheep_position[i] = [0,i*2]
        self.wolf_position = [7,3]
        self.set_state()
    def set_state(self):
        for i in range(0,64):
            self.state[i] = 0
        for i in range(0,4):
            state_index = self.get_state_index(self.sheep_position[i])
            self.state[state_index] = 1
        state_index = self.get_state_index(self.wolf_position)
        self.state[state_index] = -1
    def get_state_index(self, row_column_location):
        index = row_column_location[0]*8 + row_column_location[1]
        return index

    def is_valid_move(self, move_location):
        if move_location[0] < 0 or move_location[0] >= 8:
            return False
        if move_location[1] < 0 or move_location[1] >= 8:
            return False
        if move_location[0] == self.wolf_position[0] and move_location[1] == self.wolf_position[1]:
            return False
        for one_sheep_position in self.sheep_position:
            if move_location[0] == one_sheep_position[0] and move_location[1] == one_sheep_position[1]:
                return False
        return True
    '''
    for the sheeps we have 8 actions, one for each sheep
    '''
    def play (self, action):
        played = False
        done = True

        sheep_index = (int)(np.floor(action/2))
        direction = 0
        if action%2 == 0:
            direction = LEFT            
        else:
            direction = RIGHT            
            
        #first let's check which sheep does what action
        reward = -0.01

        sheep_current_location = self.sheep_position[sheep_index]
        sheep_next_location = sheep_current_location.copy()
        if direction == LEFT:
            sheep_next_location[0] += 1
            sheep_next_location[1] -= 1
        if direction == RIGHT:
            sheep_next_location[0] += 1
            sheep_next_location[1] += 1
        valid_move = self.is_valid_move(sheep_next_location)
        if valid_move:
            #print ('Valid Move')
            self.sheep_position[sheep_index][0] = sheep_next_location[0]
            self.sheep_position[sheep_index][1] = sheep_next_location[1]
            self.set_state()
            #We now need to check for 2 possible done conditions
            #1. The wolf can reach the last row in one move (Wolf wins)
            #2 The wolf cannot move (Sheeps win)
            up_left = [self.wolf_position[0]-1,self.wolf_position[1]-1]
            up_right = [self.wolf_position[0]-1,self.wolf_position[1]+1]
            down_left = [self.wolf_position[0]+1,self.wolf_position[1]-1]
            down_right = [self.wolf_position[0]+1,self.wolf_position[1]+1]
            if not self.is_valid_move(up_left) and not self.is_valid_move(up_right) and not self.is_valid_move(down_left) and  not self.is_valid_move(down_right):
                print ('sheeps win')
                print(np.reshape(game.state,(8,8)))
                return self.state, 10, True
            else:
                while True:
                    up_down = np.random.randint(2)
                    left_right = np.random.randint(2)
                    row_index, col_index = self.wolf_position[0], self.wolf_position[1]
                    if up_down == 0:
                        row_index -= 1
                    else:
                        row_index += 1
                    if left_right == 0:
                        col_index -= 1
                    else:
                        col_index += 1
                    if self.is_valid_move([row_index, col_index]):
                        self.wolf_position[0] = row_index
                        self.wolf_position[1] = col_index
                        self.set_state()
                        break
                wolf_wins = True
                for one_sheep_position in self.sheep_position:
                    if one_sheep_position[0] < self.wolf_position[0]:
                        wolf_wins = False
                        break
                if wolf_wins:
                    print ('wolf wins')
                    print(np.reshape(game.state,(8,8)))
                    return self.state, -10, True
                       
                        
        else:
            #print ('Invalid Move')
            reward = 0
        '''
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
        '''
        
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


#Unit Tests
game = SheepAndWolfGame()
game.reset()
print(np.reshape(game.state,(8,8)))
for j in range(0,8):
    for i in range(0,8):
        game.play(i)
print(np.reshape(game.state,(8,8)))
