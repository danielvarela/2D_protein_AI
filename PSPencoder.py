import random
import numpy as np
from Lattice2D import Lattice2D

class PSPencoder():
    def __init__(self, seq):
        self.seq = seq
        self.size = (len(seq) - 1) * 4
        self.env = Lattice2D(self.seq)
        
    def ind_size(self):
        return self.size

    def from_double_to_action(self, OldValue):
        NewValue = (((OldValue - -0.99) * (3.9 - 0)) / (0.99 - -0.99)) + 0
        return int(NewValue)

    def from_array_to_action(self, X):
        return X.index(min(X))
    
    def to_actions(self, ind):
        actions = []
        l = ind
        n = 4
        new_chunks = [l[i:i + n] for i in range(0, len(l), n)]
        for x in new_chunks:
            actions.append(self.from_array_to_action(x))
        return actions
            
    def render(self, ind):
        actions = self.to_actions(ind)
        state, reward, energy_info = self.env._compute_state(actions)
        self.env.render(actions)
        return reward
 
    def run(self, ind):
        actions = self.to_actions(ind)
        state, reward, energy_info = self.env._compute_state(actions)
        return reward
 
class PSPencoderSimple():
    def __init__(self, seq):
        self.seq = seq
        self.size = len(seq) - 1
        self.env = Lattice2D(self.seq)
        
    def ind_size(self):
        return self.size

    def from_double_to_action(self, OldValue):
        NewValue = (((OldValue - -0.99) * (3.9 - 0)) / (0.99 - -0.99)) + 0
        return int(NewValue)
        
    def to_actions(self, ind):
        actions = []
        for x in ind:
            actions.append(self.from_double_to_action(x))
        return actions
            
    def render(self, ind):
        actions = self.to_actions(ind)
        state, reward, energy_info = self.env._compute_state(actions)
        self.env.render(actions)
        return reward
 
    def run(self, ind):
        actions = self.to_actions(ind)
        state, reward, energy_info = self.env._compute_state(actions)
        return reward
 
