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
from agents.greedy_agent import NNAgent

class NeuralNetworkPlayer():
    def __init__(self, env, seq, nn):
        self.env = env
        self.agent = NNAgent(nn)
        self.seq = seq
        
    def run(self, n_episodes=1):
        training_data = []
        accepted_scores = []    
        scores = []
        choices = []
        for each_game in range(n_episodes+1):
            self.env.reset()
            score = 0
            prev_obs = []
            for step_index in range( 1, len(self.seq) -1 ):
                if (np.random.randint(0,n_episodes) < 1):
                    action, observation = self.agent.choose_action(self.env, step_index)
                    action = np.random.randint(0,4)
                    observation = [9] * 8
                else:
                    action, observation = self.agent.choose_action(self.env, step_index)
                choices.append(action)
                new_observation, reward, done, info = self.env.step(action, step_index)
                prev_obs = new_observation
                score = reward               
                if done:
                    break
            scores.append(score) 
        return scores, choices
