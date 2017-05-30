import gym
import numpy as np
import random
from statistics import median, mean
from collections import Counter

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam


env = gym.make('MountainCar-v0')
env.reset()
N = 100
explorationRate = 0.2

def pop():
    accepted_scores = []
    training_data = []
    training_data_new = []
    scores = []

    for i_episode in range(100):
        score = 0
        game_memory = []
        prev_observation = []
        observation = env.reset()
        for i in range(200):
            #env.render()
            if i == 0:
                action = env.action_space.sample()

            else:
                if random.random() < explorationRate:
                    action = env.action_space.sample()
                    print("randomActino")
                else:
                    if (observation[1] < 0):
                        action = 0

                    elif (observation[1] > 0):
                        action = 2
                    else:
                        action = 1

            #action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation

            score += reward
            if observation[0] > 0.5:
                break

        if score > -200:
            #print("made it")
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])

            for i in range(200, 0, -1):
                training_data_new.append(training_data[len(training_data)-i])
        env.reset()

    scores.append(score)
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))


    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    return training_data


training_data = pop()
X = np.array([i[0] for i in training_data])
y = [i[1] for i in training_data]



def test(_lr, _epochs, whatModel):
    if whatModel == 1:
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=2))
        model.add(Dropout(0.8))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.8))
        model.add(Dense(3, activation='softmax'))

        sgd = SGD(lr=_lr, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        model.fit(X, y, epochs=_epochs)

    if whatModel == 2:
        model = Sequential()
        model.add(Dense(_epochs, input_dim=2, activation='relu'))
        model.add(Dense(_epochs, activation='relu'))
        model.add(Dense(3))
        sgd = SGD(lr=_lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

        model.fit(X, y, epochs=_epochs)


    env = gym.make('MountainCar-v0')
    env.reset()
    scores = []
    choices = []
    learningRate = []
    epochstocomplere = []
    for each_game in range(10):
        score = 0
        game_memory = []
        env.reset()
        for _ in range(200):
            #env.render()

            if (_) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, 2)))

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if new_observation[0] > 0.5:
                break

        scores.append(score)

    print('Average Score:', sum(scores) / len(scores))
    print('choice2:{} choice 1:{}  choice 0:{}'.format(choices.count(2) / len(choices), choices.count(1) / len(choices), choices.count(0) / len(choices)))


#test(0.02, 20, 1)
test(0.005, 30, 2)


'''for epochs in range(10, 20, 2):
    for learn in range(100, 1000, 1):
        test(1/learn, epochs)'''
