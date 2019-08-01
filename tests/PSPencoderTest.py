import random
import numpy as np
from threading import Thread
from visual_utils.bokeh_visualizer import Visualizer
import gym

from gym import spaces
from agents.energy_sampler import EnergySampler
from agents.greedy_agent import NNAgent
from agents.greedy_agent import NNGridAgent
from gym_lattice.envs import Lattice2DEnv
from agents.keras_nn import FFmodel
from agents.NeuralNetworkOperator import ExtensiveLatticeNeuralNetworkOperator
from agents.NeuralNetworkOperator import NeuralNetworkOperator
from lattices.Lattice2D import Lattice2D
from PSP.PSPencoder import PSPencoder

class PSPencoderTest():
    def init_encoder(self):
        seq = "HPPH"
        encoder = PSPencoder(seq)
        ind_len = encoder.ind_size()
        assert(ind_len == (len(seq) - 1))

    def test_to_actions(self):
        seq = "HPPH"
        encoder = PSPencoder(seq)
        test_actions = [-0.6] * (len(seq) - 1)
        actions = encoder.to_actions(test_actions)
        print(actions)


test = PSPencoderTest()
test.init_encoder()
test.test_to_actions()
