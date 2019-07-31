import random
import numpy as np
from threading import Thread
from bokeh_visualizer import Visualizer
import gym
from gym import spaces
from agents.energy_sampler import EnergySampler
from agents.greedy_agent import NNAgent
from agents.greedy_agent import NNGridAgent
from keras_nn import FFmodel
from NeuralNetworkOperator import ExtensiveLatticeNeuralNetworkOperator
from NeuralNetworkOperator import NeuralNetworkOperator
from Lattice2D import Lattice2D

class NeuralNetworkGridTest():
    def __init__(self):
        self.seq = "HPPHPPH"
        self.model = FFmodel(100)
        self.operator = NeuralNetworkOperator(self.seq, "grid")
        self.env = Lattice2D(self.seq)
        
    def check_env(self):
        print("init env")
        current_sol = [1]*(len(self.seq)-1)
        state, reward, energy_info = self.operator.env._compute_state(current_sol)
        print("reward ", reward)
        self.operator.env.render(self.operator.current_sol)
        
    def check_choose_best_move(self):
        print("init check best move")
        self.operator.env.render(self.operator.current_sol)
        print("select best move")
        idx = 1
        action, rewards = self.operator.best_next_move(idx, self.model)
        #action = 0
        print("selected %s from rewards %s " % (action, rewards))
        print("sequence is not modified ")
        self.operator.env.render(self.operator.current_sol)
        # assert current sol is not modified with best_next_move 
        
    def check_step_move(self):
        print("perform the recommended action ")
        idx = 1
        action = 3
        self.operator.step(idx, action)
        self.operator.env.render(self.operator.current_sol)
        # assert current sol is modified with action at the idx position 
        
    def check_run(self):
        print("init run")
        self.operator.run(self.model)
        self.operator.env.render(self.operator.current_sol)
        print("end")
        
test = NeuralNetworkGridTest()
test.check_env()
test.check_choose_best_move()
test.check_step_move()
test.check_run()
