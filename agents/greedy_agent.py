import gym
from gym import spaces
import numpy as np
from agents.energy_sampler import EnergySampler
from agents.energy_sampler import EnergySamplerSimple
from agents.energy_sampler import GridSampler

class GreedyAgent():
    def __init__(self):
        self.e = EnergySampler() 
    def choose_action(self, env, res_pos):
        rewards = self.e.get_current_observation(env, env.actions, res_pos)
        #if (np.random.randint(0,10) < 3) :
        next_rewards = rewards[:len(rewards)//2]
        min_rewards = [i for i,x in enumerate(next_rewards) if x == min(next_rewards)]
        choice = min_rewards[np.random.randint(0, len(min_rewards))]
        #exit(1)
        return choice, rewards

class GreedyAgentSimple():
    def __init__(self):
        self.e = EnergySampler() 
    def choose_action(self, env, res_pos):
        rewards = self.e.get_current_observation(env, env.actions, res_pos)
        #if (np.random.randint(0,10) < 3) :
        min_rewards = [i for i,x in enumerate(rewards) if x == min(rewards)]
        choice = min_rewards[np.random.randint(0, len(min_rewards))]
        return choice, rewards
               
class NNAgent():
    def __init__(self, input_nn):
        self.trained_nn = input_nn
        self.e = EnergySampler() 
    def choose_action(self, env, actions, res_pos):
        obs = self.e.get_current_observation(env, actions, res_pos)
        i = np.reshape(obs, (1,8))
        action = np.argmax(self.trained_nn.predict(i)[0])
        return action, obs

class NNAgentSimple():
    def __init__(self, input_nn):
        self.trained_nn = input_nn
        self.e = EnergySamplerSimple() 
    def choose_action(self, env, actions, res_pos):
        obs = self.e.get_current_observation(env, actions, res_pos)
        i = np.reshape(obs, (1,4))
        action = np.argmax(self.trained_nn.predict(i)[0])
        return action, obs


class NNGridAgent():
    def __init__(self, input_nn):
        self.trained_nn = input_nn
        self.e = GridSampler() 
    def choose_action(self, env, actions, res_pos):
        obs = self.e.get_current_observation(env,actions, res_pos)
        i = np.reshape(obs, (1,100))
        action = np.argmax(self.trained_nn.predict(i)[0])
        return action, obs


class DatasetNNAgent():
    def __init__(self, input_nn):
        self.nn_agent = NNAgent(input_nn) 
        self.e = EnergySampler() 
        
    def choose_action(self, env, res_pos):
        if (np.random.randint(0,100) < 10):
            result = (np.random.randint(0,4), self.e.get_current_observation(env, res_pos))
        else:
            result = self.nn_agent.choose_action(env,res_pos)
        return result
