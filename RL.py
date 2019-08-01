import gym
from gym_lattice.envs import Lattice2DEnv
from gym import spaces
import operator
import numpy as np
from collections import OrderedDict

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from gym_lattice.envs import RenderProt
from sklearn.preprocessing import StandardScaler
from agents.energy_sampler import EnergySampler
from agents.greedy_agent import GreedyAgent
from agents.greedy_agent import DatasetNNAgent
from training.dataset_builder import DatasetBuilder
from predict.neural_network_player import NeuralNetworkPlayer

#np.random.seed(42)
np.random.random()

#seq = 'HHHHHPHHHHHHPHHHHPHH' # Our input sequence
seq = 'HPHPPHHPHPPHPHHPPHPH' # Our input sequence
action_space = spaces.Discrete(4) # Choose among [0, 1, 2 ,3]
env = Lattice2DEnv(seq)

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='sigmoid'))
    model.add(Dense(52, activation='sigmoid'))
    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(loss='mse', optimizer=Adam())
    #model.compile(loss='mse', optimizer="rmsprop")
    return model

def train_model(current_model, training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    #model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    current_model.fit(X, y, epochs=2000, verbose=0)
    return current_model

def print_stats(choices):
    print("len choices dataset " , len(choices))
    print('choice 0:{}  choice 1:{}'.format(choices.count(0)/len(choices),choices.count(1)/len(choices)))
    print('choice 2:{}  choice 3:{}'.format(choices.count(2)/len(choices),choices.count(3)/len(choices)))

def print_trainingdata_stats(training_data_in, accepted_scores = []):
    observations = [tuple(x[0]) for x in training_data]
    dict_obs = OrderedDict()
    for o in observations:
        if (o in dict_obs.keys()):
            v = dict_obs[o]
            dict_obs.update({o : v + 1})
        else:
            dict_obs.update({o : 1})
        
    print("len observations ", len(dict_obs.keys()))
    print("unique observations " ,len(set(dict_obs.keys())))
    choices = [ x[1].index(max(x[1])) for x in training_data]
    print_stats(choices)
    print("accepted scores " , len(accepted_scores))
    print("min accepted scores " , min(accepted_scores))
    print("avg accepted scores " , np.average(accepted_scores))
    #print(accepted_scores)

def print_nnplayer_result(scores, choices):
    print(scores)
    print('Best score: ', min(scores))
    print('Average Score:', sum(scores)/len(scores))
    print_stats(choices)


cycles = 20
min_limit = -8
current_agent = GreedyAgent()
best_training_data = 0
best_agent_value = 0
episodes = 50
current_model = build_model(8, 4)
best_model = current_model
current_data = []
for i in range(episodes):
    print("episode ", i)
    lim = min(min_limit, best_training_data)
    print("build data with limit " , lim)
    data_builder = DatasetBuilder(env, seq, current_agent, lim + 2)
    training_data_build, accepted_scores = data_builder.run(cycles, current_data)
    training_data_sort = sorted(training_data_build, key=lambda k : k["score"])
    accepted_scores = data_builder.get_plain_accepted_scores(training_data_sort)
    min_scores = min(accepted_scores)
    best_index = [i for i, x in enumerate(training_data_sort) if x["score"] == min_scores ]
    # current_data = [] 
    # if (len(best_index) > 3):
    #     while(len(current_data) < 3):
    #         current_data.append(training_data_sort[best_index[np.random.randint(0, len(best_index))]])
    # else:
    #     current_data = training_data_sort[:3]
    if (len(training_data_sort) > 50):
        current_data = training_data_sort[:50]
    else:
        current_data = training_data_sort
        
    training_data = data_builder.get_plain_training_data(current_data)
    accepted_scores = data_builder.get_plain_accepted_scores(current_data)
    print_trainingdata_stats(training_data, accepted_scores)
    for o, a in training_data_sort[0]["data"]:
        print("obs :", o, end =" ")
        print(a.index(max(a)))
    best_training_data = min(accepted_scores)
    print("start train model")
    backup_model = current_model
    trained_model = train_model(current_model, training_data)
    current_model = backup_model
    nn_player = NeuralNetworkPlayer(env, seq, trained_model)
    print("NN player")
    scores, choices = nn_player.run(50)
    print("agent avg scores " + str(np.average(scores)))
    #print_nnplayer_result(scores, choices)
    if min(scores) < best_agent_value:
        best_agent = current_agent
        best_agent_value = min(scores)
        best_model = trained_model
        current_model = trained_model
        print("new best model found " , best_agent_value)
    else:
        print("current trained model ", min(scores) ) 
    if (best_agent_value < best_training_data):
        current_agent = DatasetNNAgent(trained_model)
    if (min(scores) > -1):
        print("otra vez " , best_agent_value)
        print(env.actions)
        env.render()
        
    if(best_agent_value <= -10):
        print("modelo entrenado!! ", best_agent_value)
        break
