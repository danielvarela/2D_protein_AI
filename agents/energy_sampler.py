import gym
from gym import spaces
import numpy as np


from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from agents.RenderProt import RenderProt
from sklearn.preprocessing import StandardScaler

class EnergySamplerSimple():
    def get_current_observation(self, env, in_actions, res_pos):
        state, start_energy, energy_info = env._compute_state(in_actions)
        copy_actions = in_actions
        rewards = []
        action_pos = res_pos
        old_action = copy_actions[res_pos]
        actions = in_actions
        actions[action_pos] = 0
        scaler = StandardScaler()
        for try_action in range(0, 4):
            actions[action_pos] = try_action
            backup_state = state
            state, new_energy, energy_info = env._compute_state(actions)
            reward = new_energy - start_energy
            rewards.append(reward)
        rewards = scaler.fit_transform(np.reshape(rewards, (-1,1)) )
        observations = np.reshape(rewards, (1,4))[0]
        in_actions[action_pos] = old_action
        return observations

class EnergySampler():
    def compute_greedy(self, env, actions, res_pos):
        action_pos = res_pos
        state, start_energy, energy_info = env._compute_state(actions)
        rewards_next = []
        if (action_pos) < (len(env.seq) - 1):
            #if False:
            next_state = state
            #new_actions = actions + [0]
            old_action = actions[res_pos]
            new_actions = actions
            next_state, start_energy, energy_info = env._compute_state(new_actions)
            for next_action in range(0,4):
                new_actions[action_pos] = next_action
                state, new_energy, energy_info = env._compute_state(new_actions)
                reward = new_energy - start_energy
                rewards_next.append(reward)

            actions[res_pos] = old_action
        else:
            rewards_next = [0] * 4
        return rewards_next
 
    def get_current_observation(self, env, in_actions, res_pos):
        state, start_energy, energy_info = env._compute_state(in_actions)
        copy_actions = in_actions
        rewards = []
        action_pos = res_pos
        old_action = copy_actions[res_pos]
        #state = env._compute_state(state, actions)
        actions = in_actions
        actions[action_pos] = 0
        scaler = StandardScaler()
        greedy_rewards = []
        for try_action in range(0, 4):
            actions[action_pos] = try_action
            backup_state = state
            state, new_energy, energy_info = env._compute_state(actions)
            reward = new_energy - start_energy
            rewards_next = self.compute_greedy(env, actions, action_pos + 1)
            rewards.append(reward)
            greedy_rewards.append(min(rewards_next))
        rewards = scaler.fit_transform(np.reshape(rewards, (-1,1)) )
        greedy_rewards = scaler.fit_transform(np.reshape(greedy_rewards, (-1,1)) )
        rewards = np.concatenate( (rewards ,greedy_rewards ), axis= 0  )
        observations = np.reshape(rewards, (1,8))[0]
        in_actions[action_pos] = old_action
        return observations

class GridSampler():
    def get_current_observation(self, env, in_actions, res_pos):
        backup_env = env
        copy_actions = in_actions
        rewards = []
        action_pos = res_pos
        old_action = copy_actions[-1]
        scaler = StandardScaler()
        state, start_energy, energy_info = env._compute_state(in_actions)
        obs_grid = env._get_grid(state, res_pos)
        observations = obs_grid.reshape(1,100)
        env = backup_env
        return observations
