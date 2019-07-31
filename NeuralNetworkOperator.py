
import gym
from gym import spaces
from agents.energy_sampler import EnergySampler
from agents.energy_sampler import EnergySamplerSimple
from agents.greedy_agent import NNAgent
from agents.greedy_agent import NNAgentSimple
from agents.greedy_agent import NNGridAgent
from Lattice2D import Lattice2D
from keras_nn import FFmodel


class NeuralNetworkOperator():
    def __init__(self, seq, strategy = "nn_operator"):
        self.seq = seq
        self.action_space = spaces.Discrete(4) # Choose among [0, 1, 2 ,3]
        self.env = Lattice2D(seq)
        self.current_sol = [1] * (len(self.seq) - 1)
        self.current_state = self.env._compute_state(self.current_sol)
        self.strategy = strategy
        
    def best_next_move(self, res_pos, model):
        nn_agent = NNAgent(model)
        return nn_agent.choose_action(self.env, self.current_sol, res_pos)
    
    def reset_sol(self):
        self.current_sol = [1] * (len(self.seq) - 1)
        self.current_state = self.env._compute_state(self.current_sol)
        
    def step(self, res_pos, action):
        self.current_sol[res_pos] = action
        state, reward, energy_info = self.env._compute_state(self.current_sol)
        return state, reward, energy_info["done"], energy_info
        
    def render(self, agent):
        reward = 0
        self.reset_sol()
        state = self.current_state
        for d in range(0, 1):
            step_reward = reward 
            for idx in range(1, len(self.seq) - 1 ):
                action, rewards = self.best_next_move(idx, agent)
                state, reward, done, info = self.step(idx, action)
                if done:
                    break
        state, final_reward, energy_info = self.env._compute_state(self.current_sol)
        print(" > FREE GIBBS ENERGY SOLUTION : " , final_reward)
        self.env.render(self.current_sol)
        
    def run(self, agent):
        self.reset_sol()
        reward = 0
        final_rewards = []
        for d in range(0,1):
            step_reward = reward
            for idx in range(1, len(self.seq) - 1 ):
                action, rewards = self.best_next_move(idx, agent)
                #print(self.current_sol)
                self.current_state, t_reward, done, info = self.step(idx, action)
                #print("at %s : %s do %s "% (idx, rewards, action))
                if (action != self.current_sol[idx-1]):
                    t_reward = t_reward - 0.003
                if (t_reward < 0):
                    t_reward = t_reward * 100
                if (t_reward >= 0):
                    t_reward = t_reward + 0.01
                reward += t_reward
                if info["collisions"] > 0:
                    break
                if done:
                    break

        #print(final_rewards)
        #print(reward)
        return reward



class ExtensiveLatticeNeuralNetworkOperator(NeuralNetworkOperator):
    def __init__(self, seq, strategy = "nn_operator_ext"):
        self.seq = seq
        self.action_space = spaces.Discrete(4) # Choose among [0, 1, 2 ,3]
        self.env = Lattice2D(seq)
        self.current_sol = [1]
        self.current_state = self.env._compute_state(self.current_sol)
        self.strategy = strategy
      
    def best_next_move(self, res_pos, model):
        self.current_sol.append(1)
        nn_agent = NNAgentSimple(model)
        return nn_agent.choose_action(self.env, self.current_sol, res_pos)
 
    def reset_sol(self):
        self.current_sol = [1]
        self.current_state = self.env._compute_state(self.current_sol)
        
    def step(self, res_pos, action):
        self.current_sol[res_pos] = action
        state, reward, energy_info = self.env._compute_state(self.current_sol)
        return state, reward, energy_info["done"], energy_info
        
    def render(self, agent):
        reward = 0
        self.reset_sol()
        state = self.current_state
        for idx in range(1, len(self.seq) - 1 ):
            action, rewards = self.best_next_move(idx, agent)
            state, reward, done, info = self.step(idx, action)
            if done:
                break
        state, final_reward, energy_info = self.env._compute_state(self.current_sol)
        print(" > FREE GIBBS ENERGY SOLUTION : " , final_reward)
        self.env.render(self.current_sol)
        
    def run(self, agent):
        self.reset_sol()
        reward = 0
        final_rewards = []
        step_reward = reward
        for idx in range(1, len(self.seq) - 1 ):
            action, rewards = self.best_next_move(idx, agent)
            #print(self.current_sol)
            self.current_state, t_reward, done, info = self.step(idx, action)
            #print("at %s : %s do %s "% (idx, rewards, action))
            if not(action in self.current_sol[:idx-1]):
                t_reward = t_reward - 0.003
            if (t_reward < 0):
                t_reward = t_reward * 10
            if (t_reward >= 0):
                t_reward = t_reward + 0.01
            reward += t_reward
            if info["collisions"] > 0:
                break
            if done:
                break
            
        state, final_reward, energy_info = self.env._compute_state(self.current_sol)
        reward = final_reward * 10 + reward * 0.3
        return reward


class NeuralNetworkGridOperator(NeuralNetworkOperator):
    def __init__(self, seq, strategy = "grid"):
        self.seq = seq
        self.action_space = spaces.Discrete(4) # Choose among [0, 1, 2 ,3]
        self.env = Lattice2D(seq)
        self.current_sol = [1] 
        self.current_state = self.env._compute_state(self.current_sol)
        self.strategy = strategy
        
    def best_next_move(self, res_pos, model):
        nn_agent = NNGridAgent(model)
        return nn_agent.choose_action(self.env, self.current_sol, res_pos)
 
    def reset_sol(self):
        self.current_sol = [1]
        self.current_state = self.env._compute_state(self.current_sol)
        
    def step(self, res_pos, action):
        if (len(self.current_sol) < (len(self.seq) - 1)):
            self.current_sol.append(action)
        else:
            self.current_sol[res_pos] = action
            
        state, reward, energy_info = self.env._compute_state(self.current_sol)
        return state, reward, energy_info["done"], energy_info
        
    def render(self, agent):
        reward = 0
        self.reset_sol()
        state = self.current_state
        done = False
        for i in range(0, 3):
            if done:
                break
            for idx in range(1, len(self.seq) - 1 ):
                action, rewards = self.best_next_move(idx, agent)
                state, reward, done, info = self.step(idx, action)
                if done:
                    break
        state, final_reward, energy_info = self.env._compute_state(self.current_sol)
        print(" > FREE GIBBS ENERGY SOLUTION : " , final_reward)
        self.env.render(self.current_sol)
   
    def run(self, agent):
        self.reset_sol()
        reward = 0
        final_rewards = []
        done = False
        for i in range(0, 3):
            if done:
                break
            for idx in range(1, len(self.seq) - 1 ):
                action, rewards = self.best_next_move(idx, agent)
                #print(self.current_sol)
                self.current_state, t_reward, done, info = self.step(idx, action)
                #print("at %s : %s do %s "% (idx, rewards, action))
                if (action != self.current_sol[idx-1]):
                    t_reward = t_reward - 0.003
                if (t_reward < 0):
                    t_reward = t_reward * 2
                if (t_reward >= 0):
                    t_reward = t_reward + 0.01
                reward += t_reward
                if info["collisions"] > 0:
                    done = True
                    break
                if done:
                    break

        #print(final_rewards)
        #print(reward)
        return reward
 
class NNOperatorBuilder():
    def __init__(self, seq):
        self.seq = seq

    def get(self, strategy):
        if (strategy is "nn_operator") :
            return NeuralNetworkOperator(self.seq, "nn_operator")
        if (strategy is "grid") :
            return NeuralNetworkGridOperator(self.seq, "grid")
        if (strategy is "nn_operator_ext") :
            return ExtensiveLatticeNeuralNetworkOperator(self.seq, "nn_operator_ext")
        
