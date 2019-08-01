from cost_funcs.cost_func import CostFunction
from PSP.PSPencoder import PSPencoder

class PSPFunction(CostFunction):
    def __init__(self, seq):
        self.name = "PSP function"
        self.seq = seq
        self.encoder = PSPencoder(self.seq)
        
    def size(self):
        return self.encoder.ind_size()
    
    def run(self, x):
        reward = 0
        reward = self.encoder.run(x)
        return reward

    def render(self, x):
        self.encoder.render(x)
        
 
