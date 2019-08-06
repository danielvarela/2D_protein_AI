from cost_funcs.cost_func import CostFunction

from agents.greedy_agent import NNAgent
from agents.greedy_agent import NNGridAgent
from agents.keras_nn import FFmodel
from agents.NeuralNetworkOperator import ExtensiveLatticeNeuralNetworkOperator
from agents.NeuralNetworkOperator import NeuralNetworkOperator
from agents.NeuralNetworkOperator import NeuralNetworkGridOperator
from agents.NeuralNetworkOperator import NNOperatorBuilder

class NeuralNetworkFunction(CostFunction):
    def __init__(self, seq, strategy = "nn_operator"):
        self.name = "NN function"
        self.strategy = strategy
        self.seq = seq
        if (self.strategy == "grid"):
            self.model = FFmodel(100)
        if (self.strategy == "nn_operator"):
            self.model = FFmodel(8)
        if (self.strategy == "nn_operator_ext"):
            self.model = FFmodel(4)

        self.operator = NNOperatorBuilder(self.seq).get(self.strategy)
            
    def size(self):
        return self.model.nn_connections()
    
    def run(self, x):
        #print("run " + str(len(x)) )
        self.model.set_weights(x)
        # run model from 1 to len(self.seq)
        # reward is energy value of the folded protein
        reward = 0
        reward = self.operator.run(self.model)
        return reward

    def render(self, x):
        self.model.set_weights(x)
        self.operator.render(self.model)
