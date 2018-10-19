import numpy as np
import random
import random
import time

random.seed(time.time())

#this is the current poistion on a rectangular board
#Board Layout
# X = current position
#  _ = Empty cell
# * = Blocker
# $ = Goal
'''
X _ _ _ _
_ _ _ _ _
_ _ * _ _
_ _ _ * *
_ _ _ _ $
'''
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
current_position = np.array([0,0])
height = 5
width = 5
goal_location = np.array([4,4])
barrier_locations = np.array([
    [2,2],
    [3,3],
    [3,3],
    [3,4]
])


#We do not save all possible State-Action combinations and their respective value (Q)
#We just save the ones we encounter. To bootstrap we just assume equal value (Q) for each possible Action in a state
#We select the Action randomly with the ODDS bering proportional to the current Value of each action available at this state
#Therefor we start off with an empty state-action to Value map (Dictionary)
stateToActionValuesMap = {}
stateActionHistory = []#this is an array or lisnked list of all the state actions TAKEN
np.set_printoptions(precision=2)

#scale so that the sum of all action values is 1
def scale(stateActionsValues):
    sum_of_values = 0
    for key in stateActionsValues:
        sum_of_values += stateActionsValues[key]
    for key in stateActionsValues:
        stateActionsValues[key] = 1 * stateActionsValues[key]/sum_of_values
    
def chooseAction(stateActionsValues, followBestValue=False):
    rand = random.random()
    start_value_for_key = 0
    if followBestValue:
        bestValue = None
        bestValueKey = None
        for key in stateActionsValues:
            value = stateActionsValues[key]
            if bestValue == None or value > bestValue:
                bestValue = value
                bestValueKey = key
        return bestValueKey
            
    else:
        for key in stateActionsValues:
            value = stateActionsValues[key]
            #print('Key: ', key, ' Value: ', value)
            end_value_for_key = start_value_for_key + value
            if( rand >= start_value_for_key and rand <= end_value_for_key):
                return key
            start_value_for_key = end_value_for_key

def hitBarrier(next_position):
    #check against barrier_locations
    return False

def calculateRewardAndAdjustStateActionValues():
    reward = 1
    learning_rate = 0.001
    #stateToActionValuesMap = {}
    #stateActionHistory = []#this is an array or lisnked list of all the state actions TAKEN
    historyLength = len(stateActionHistory)
    for i in range (0, historyLength):
         #print (stateActionHistory[i])
         state_key = stateActionHistory[i]['state']
         action = stateActionHistory[i]['action']
         stateActionsValues = stateToActionValuesMap[state_key]
         currentStateActionValue = stateActionsValues[action]
         punishment = historyLength * 0.01
         reward -= punishment
         change = learning_rate * reward
         if( currentStateActionValue + change > 0.001 ):
             stateActionsValues[action] += change
         scale(stateActionsValues)
         #print (currentStateActionValue)
         
    #print(historyLength)
    
         
def move(from_position, followBestValue=False):
    #Step 1: Find the Value associated with each available Action in the current State. If no such Value create a default one for each action
    #print(from_position)
    key = np.array2string(from_position)
    stateActionsValues = stateToActionValuesMap.get(key, None)
    if(stateActionsValues == None):
        stateActionsValues = {}
        stateToActionValuesMap[key] = stateActionsValues
        if (from_position[0] > 0):
            stateActionsValues[UP] = 1
        if (from_position[1] < width-1):
            stateActionsValues[RIGHT] = 1
        if (from_position[0] < height-1):
            stateActionsValues[DOWN] = 1
        if (from_position[1] > 0):
            stateActionsValues[LEFT] = 1
        scale(stateActionsValues)
    else:
        #print('Key Found: ', key)
        dummy = 1
    chosen_action = chooseAction(stateActionsValues, followBestValue)
    #print('availavle actions and Values: ', stateActionsValues)
    #we now proceed to calculate the reward for any actions in this chain.
    next_position= from_position.copy()
    if(chosen_action == UP):
        next_position[0] -= 1
    if(chosen_action == DOWN):
        next_position[0] += 1

    if(chosen_action == LEFT):
        next_position[1] -= 1
    if(chosen_action == RIGHT):
        next_position[1] += 1

    stateActionHistory.append({'state':np.array2string(from_position), 'action':chosen_action})
    #check if we hit one of the barrier_locations
    # in such a case issue negative reward. That state action should be very low value (0)
    # No progress to next cell will be made
    if (hitBarrier(next_position)):
        return(start_position)
    else:
        return next_position
        

    #Check if we got to our destination. If not then issue small negative reward for waisting steps
    #if we got to our goal we issue a positive reward.
    #we can actually just delay calculation of the final reward and only calculate at the end - and propagate down the chain
    #we should also deduct the number of nodes in the current chain times some value 
    #print(next_position, ' Prev: ', from_position)



#move(np.array([0,0]))
#move(np.array([0,0]))
#move(np.array([4,0]))
#move(np.array([0,4]))
#move(np.array([4,4]))
#for i in range(0,5):
#    for j in range (0,5):
#        move(np.array([i,j]))
def epoch(followBestValue=False):
    position = np.array([0,0])
    for i in range(0,500):
        position = move(position,followBestValue=False)
        #print(position)
        if(position[0] == 4 and position[1] == 4):
            calculateRewardAndAdjustStateActionValues()
            break

for x in range (1,10):
    for j in range(0,10000) :
        #print("epoch: ", j)
        epoch(followBestValue=False)
        historyLength = len(stateActionHistory)
        #print(historyLength)
        stateActionHistory.clear()
    epoch(followBestValue=True)
    print(historyLength)
