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


### set up environment
env = gym.make('CartPole-v0')
env.reset()


#load numpy array from file
training_data = np.load('saved-data-gen.npy')

### neural network design
def neural_network_model(input_size):

    #input layer
    network = input_data(shape=[None, input_size, 1], name='input')

    # hidden layers
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

    # output layer
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


### train model and fit
def train_model(training_data, model=False):
    # extract observations
    X = np.array([i[0] for i in training_data]).reshape(-1, training_data[0][0].size, 1)
    # extract actions
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=X[0].size)

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


model = train_model(training_data)


### run game x times with predictions from trained model
scores = []
choices = []
obs = []

second_gen_all_choices = []
second_gen_all_obs = []
for each_game in range(output_games):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()


    # per frame
    for _ in range(200):

        # render the game, comment this out to speed up the process
        #env.render()

        # start with a random action in 1st frame
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        # continue with predicted actions
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        obs.append(prev_obs)
        game_memory.append([new_observation, action])

        score += reward
        if new_observation[0] > 2.39 or new_observation[0] < -2.39 \
                or new_observation[2] > 0.19 or new_observation[2] < -0.19:
            break
    second_gen_all_choices.append(choices)
    second_gen_all_obs.append(obs)
    scores.append(score)

#env.close()
#print(scores)
#print("best samples mean: ", np.mean(scores_values))
#print("length: ", len(scores_values))

#plt.hist(scores, 50, normed=1)
#plt.show()

### SECOND GENERATION


### format the training data
# iterate each sample
second_gen_training_data = []
for i in range(len(second_gen_all_choices)):
    for j in range (len(second_gen_all_choices[i])):

        if second_gen_all_choices[i][j] == 1:
            output = [0, 1]
        elif second_gen_all_choices[i][j] == 0:
            output = [1, 0]

        # previous observations plus actions, stored in training data
        second_gen_training_data.append([second_gen_all_obs[i][j], output])

### save data from sample, uncomment to overwrite file
training_data_save = np.array(second_gen_training_data)
np.save('saved-data-gen.npy', training_data_save)

env.close()
print(scores)

plt.hist(scores, 50, normed=1)
plt.show()




