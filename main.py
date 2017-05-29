import random
import numpy as np
import gym
import tflearn
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#learning rate for model
LR = 0.001

#global variables
best_samples = []
best_samples_observations = []
scores_values = []
training_data = []


### algorithm variables
number_of_generations = 3
sample_size = 400
best_sample_size = 20
output_games = 10


### function for creating new generations
def create_clones(samples_to_clone, current_best_index):

    # make all rows identical to the best sample so far
    copied_rand_numbers = samples_to_clone
    for i in range(copied_rand_numbers[0, :].size):
        copied_rand_numbers[i, :] = samples_to_clone[current_best_index, :]

    # change some numbers if a certain probability is met
    for i in range(copied_rand_numbers[0,:].size):
        for j in range(copied_rand_numbers[:,0].size):
            probability = random.randrange(0, 10)
            if probability > 7:
                if copied_rand_numbers[j, i] == 0:
                    copied_rand_numbers[j, i] = 1
                else:
                    copied_rand_numbers[j, i] = 0

    #return the new, "mutated" array
    return copied_rand_numbers


### function, runs the game and generates new observations
def run_game(samples):

    global best_score
    global best_score_index

    for i in range(samples[:, 0].size):
        env.reset()
        score = 0
        prev_observation = []
        for j in range(samples[0, :].size):
            action = samples[i, j]
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                obs[i, j] = prev_observation
            prev_observation = observation
            score += reward
            if done:
                break
        if score > best_score:
            best_score = score
            best_score_index = i
    return

### the main loop, iterates for every sample size to be generated
for _ in range(best_sample_size):

    # set up initial environment
    env = gym.make('CartPole-v0')
    env.reset()
    best_score = 0
    best_score_index = 0
    #set up the observations
    obs = np.zeros((sample_size, 200, 4))

    # initial population
    rand_numbers = np.zeros((sample_size, 200), dtype=np.int)
    #inital randomness
    for i in range(rand_numbers[:,0].size):
        for j in range(rand_numbers[0, :].size):
            rand_numbers[i,j] = random.randrange(0,2)

    #prepare an array for new generations
    generations = [None]*number_of_generations
    generations[0] = rand_numbers

    for x in range(len(generations)):
        run_game(generations[x])
        if x < len(generations)-1:
            generations[x+1] = create_clones(generations[x], best_score_index)
        else:
            best_samples.append(generations[x][best_score_index,:])
            best_samples_observations.append(obs[best_score_index,:])
            scores_values.append(best_score)



### format the training data
# iterate each sample
for i in range(len(best_samples)):
    # iterate each action
    for j in range(len(best_samples[i])):
        if best_samples[i][j] == 1:
            output = [0, 1]
        elif best_samples[i][j] == 0:
            output = [1, 0]

        # previous observations plus actions, stored in training data
        training_data.append([best_samples_observations[i][j], output])



### save data from sample, uncomment to overwrite file
#training_data_save = np.array(training_data)
#np.save('saved-data.npy', training_data_save)

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, training_data[0][0].size, 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=X[0].size)

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


model = train_model(training_data)

scores = []
choices = []
for each_game in range(output_games):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(200):
        env.render()

        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done: break

    scores.append(score)

env.close()
print(scores)
print("best samples mean: ", np.mean(scores_values))
print("length: ", len(scores_values))

plt.hist(scores, 50, normed=1)
plt.show()




