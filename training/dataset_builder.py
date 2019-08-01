import gym
from gym_lattice.envs import Lattice2DEnv
from gym import spaces
import numpy as np

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from gym_lattice.envs import RenderProt
from sklearn.preprocessing import StandardScaler
from agents.energy_sampler import EnergySampler
from agents.greedy_agent import GreedyAgent

class DatasetBuilder():
    def __init__(self, env, seq, agent = GreedyAgent(), limit = -9):
        self.env = env
        self.agent = agent
        self.seq = seq
        self.limit = limit
        
    def run(self, n_episodes, old_data = []):
        training_data = old_data
        accepted_scores = []    
        while(len(accepted_scores) < 100):
            for i_episodes in range(n_episodes+1):
                if (i_episodes % n_episodes/10) == 0:
                    print("training episode " + str(i_episodes) )
                    print(len(training_data))
                    print(len(accepted_scores))
                best_reward = 0
                self.env.reset()
                game_memory = []
                previous_observation = []
                score = 0
                last_reward = 0
                last_position = 0
                for idx in range(1, len(self.seq) - 1 ):
                    last_position = idx
                    sent_env = self.env
                    before_len = len(self.env.actions)
                    action, observations = self.agent.choose_action(sent_env, idx)
                    #print("select " + str(action) + " at " + str(idx) )
                    #print("observations ", observations)
                    obs, reward, done, info = self.env.step(action, idx)
                    game_memory.append([observations, action])
                    previous_observation = observations
                    score = reward
                    if (reward < best_reward):
                        best_reward = reward
                    if done:
                        break
                    else:
                        last_reward = reward
                if (score < self.limit) :
                    accepted_scores.append(score)
                    training_data_tmp = []
                    for data in game_memory:
                        if data[1] == 0:
                            output = [1,0,0,0]
                        if data[1] == 1:
                            output = [0,1,0,0]
                        if data[1] == 2:
                            output = [0,0,1,0]
                        if data[1] == 3:
                            output = [0,0,0,1]
                        training_data_tmp.append([data[0], output])
                    training_data.append({"score" : score , "data" : training_data_tmp})
                self.limit = self.limit + 0.5
        return training_data, accepted_scores

    def get_plain_training_data(self, training_data_in):
        t_data = [x["data"] for x in training_data_in]
        training_data = []
        for t in t_data:
            for d in t:
                training_data.append(d)
        return training_data
    
    def get_plain_accepted_scores(self, training_data_in):
        return [x["score"] for x in training_data_in]
   
