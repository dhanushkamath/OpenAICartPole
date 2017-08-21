import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from time import sleep
from tqdm import tqdm
LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 100000

# IN THE CARTPOLE EXAMPLE THE OBSERVATION IS NOT THE PIXEL DATA. The observation is cart position,pole position etc
# But, in some games, the observation can be raw pixel data, so use CONVNETS in those games.

def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break

# Generating the training data
# Training data is generated randomly by taking random actions.
# Any game is taken as a training data, if the random actions resulted in scoring above 50
def initial_population():
    training_data = []
    scores = []
    accepted_scores =[]

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2) # you can use the action_space.sample(), but check documentation first
            observation, reward, done, info = env.step(action)


            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0],output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print("Average accepted score : ", mean(accepted_scores))
    print("Median accepted scores : ", median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

# Run initial_population() to generate the numpy file of training data


def neural_network_model(input_size):
    network = input_data(shape = [None,input_size, 1], name ='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.7)
    # 0.8 is actually the keep rate. so 0.8 means 0.2 dropout (not sure), just google first.

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.7)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.7)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.7)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.7)


    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name = 'targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data,model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]


    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input':X},{'targets':y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id ='openaistuff-1')

    return model

#training_data = initial_population()
#model = train_model(training_data)
#model = model.save("saved_model/OpenAI-2.model")

model = neural_network_model(4) # 4 is the input size in this case
model.load("saved_model/OpenAI-2.model")

scores = []
choices = []

for each_game in tqdm(range(30)):
    score = 0
    game_memory =[]
    prev_obs = []
    env.reset()
    for _ in range(goal_steps*2):
        env.render()
        if len(prev_obs) == 0: # Coz in the first frame, we are not sure of what move to make
            action = random.randrange(0,2) # thats why we use random

        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward

        sleep(0.02) # Introducing a lag to visualize properly
        if done:
            break
    print("\nScore of Game " + str(each_game + 1) + " is : " + str(score))
    scores.append(score)

print('Average Score', sum(scores)/len(scores))
print('Max score : ', max(scores))
print('Choice 1: {}, Choice 2 :{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))