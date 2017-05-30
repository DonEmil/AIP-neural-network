import gym
import numpy as np
import random
from statistics import median, mean
from collections import Counter

from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import SGD


env = gym.make('MountainCar-v0')
env.reset()

#This is used to take random actions.
explorationRate = 0.5

def pop():
    #These will be used to store scores.
    accepted_scores = []
    training_data = []
    #This was one used for testing where the cart only ran random actions
    training_data_new = []
    scores = []

    for i_episode in range(100):
        score = 0
        game_memory = []
        prev_observation = []
        observation = env.reset()
        for i in range(200):
            #env.render()
            #The first action is random
            if i == 0:
                action = env.action_space.sample()

            else:
                #Explore 50 % of the times
                if random.random() < explorationRate:
                    action = env.action_space.sample()
                #Else move according to the velocity of the cart.
                else:
                    if (observation[1] < 0):
                        action = 0

                    elif (observation[1] > 0):
                        action = 2
                    else:
                        action = 1

            #action = env.action_space.sample()

            #Get the location, reward and info based on the action that is being made.
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                #Store the last observation to use for neural network
                game_memory.append([prev_observation, action])
            prev_observation = observation

            #Save the score note if score reaches -200 the game is done
            score += reward
            if done:
                break

        if score > -200:
            #If the cart reached the goal use the data.
            accepted_scores.append(score)
            for data in game_memory:
                #Append its action to the training data[1}
                if data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])

            #This was used for random action testing. If the cart completed ot should only save the last 200 actions and observations.
            '''for i in range(200, 0, -1):
                training_data_new.append(training_data[len(training_data)-i])'''
        env.reset()

    #Get the averages to compare
    scores.append(score)
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))


    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    return training_data

#Store the training data from pop(). And split it up into 2 arrays. X being ( , 2) observations and y being ( , 3) possible action
training_data = pop()
X = np.array([i[0] for i in training_data])
y = [i[1] for i in training_data]



def test(_lr, _epochs, hidden_layers, whatModel):
    #The two models were used for testing a good outcome. The second model worked best overall so we deleted the first.

    if whatModel == 2:
        model = Sequential()
        model.add(Dense(hidden_layers, input_dim=2, activation='relu'))
        model.add(Dense(hidden_layers, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        sgd = SGD(lr=_lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

        model.fit(X, y, epochs=_epochs)
        model.save("model")



    env = gym.make('MountainCar-v0')
    env.reset()
    scores = []
    choices = []

    #Run the game 10 times
    for each_game in range(10):
        score = 0
        game_memory = []
        env.reset()
        for _ in range(200):
            #env.render()

            #First action is random
            if (_) == 0:
                action = env.action_space.sample()
            else:
                #Second action is based on the policy created by the fully connected layer.
                action = np.argmax(model.predict(prev_obs.reshape(-1, 2)))

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break

        scores.append(score)

    #Save and compare the score
    print('Average Score:', sum(scores) / len(scores))
    print('choice2:{} choice 1:{}  choice 0:{}'.format(choices.count(2) / len(choices), choices.count(1) / len(choices), choices.count(0) / len(choices)))


#Same model as in test(). This was used to test additional runthrough of the data. Didn't improve results
def createModel(_lr, _epochs, hidden_layers, whatModel):
    if whatModel == 2:
        model = Sequential()
        model.add(Dense(hidden_layers, input_dim=2, activation='relu'))
        model.add(Dense(hidden_layers, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        sgd = SGD(lr=_lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

        model.fit(X, y, epochs=_epochs)
        return model

#similar to pop(). it runs through making choices based on model.predict and saves them to run through createModel
def episode(model):
    accepted_scores2 = []
    training_data2 = []
    scores2 = []

    for i_episode in range(100):
        score = 0
        game_memory2 = []
        prev_obs = env.reset()
        for i in range(200):
            # env.render()
            if (_) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, 2)))

            accepted_scores = []
            training_data = []
            training_data_new = []
            scores2 = []

            observation, reward, done, info = env.step(action)
            if len(prev_obs) > 0:
                game_memory2.append([observation, action])
            prev_obs = observation

            score += reward
            if done:
                break

        if score > -130:
            print(score)
            accepted_scores2.append(score)
            for data in game_memory2:
                if data[1] == 0:
                    output2 = [1, 0, 0]
                elif data[1] == 1:
                    output2 = [0, 1, 0]
                elif data[1] == 2:
                    output2 = [0, 0, 1]
                training_data2.append([data[0], output2])

        env.reset()

    scores2.append(score)
    print('Average accepted score:', mean(accepted_scores2))
    print('Median score for accepted scores:', median(accepted_scores2))
    print(Counter(accepted_scores2))
    print(training_data2)

    training_data2_save = np.array(training_data2)
    np.save('saved.npy', training_data2_save)
    return training_data2

#model = createModel(0.01, 40, 100, 2)

#training_data2 = episode(model)

#X2 = np.array([i[0] for i in training_data2])
#y2 = [i[1] for i in training_data2]

#test(0.02, 20, 1)
#0.002, 40, 40, 2 for exploratory 0.2
#test(0.01, 40, 100, 2) for exploratory 0.5

#after some testing with different parameters, these seemed to give stable results
test(0.01, 40, 100, 2)




'''
Example
Average accepted score: -174.64102564102564


to

6772/6772 [==============================] - 0s - loss: 0.1668 - acc: 0.6692
Average Score: -124.5
choice2:0.7293172690763052 choice 1:0.005622489959839358  choice 0:0.26506024096385544

'''