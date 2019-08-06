import pygmo as pg
import numpy as np
from cost_funcs.cost_func import CostFunction
from cost_funcs.psp_func import PSPFunction

class PSPFitness(CostFunction):
    def __init__(self, seq):
        self.name = "PSP Fitness"
        self.seq = seq
        self.psp_function = PSPFunction(self.seq)
   
    def fitness(self, x):
        return [self.psp_function.run(np.array(x))]

    def get_bounds(self):
        low_bound = [-1] * self.psp_function.size() 
        upper_bound = [1] * self.psp_function.size() 
        return (low_bound, upper_bound)
    
def main():
    seq = 'HPHPHPHPPHPH' # Our input sequence

    # 1 - Instantiate an object of the PSP fitness function(HP energy function)
    # with the input sequence
    # (user defined problem).
    prob_psp = pg.problem(PSPFitness(seq))
    
    # 2 - Instantiate a pygmo algorithm
    algo_psp = pg.algorithm(pg.de(gen=10))
    algo_psp.set_verbosity(1)

    # 3 - Instantiate a pygmo archipelago
    archi = pg.archipelago(n=4,algo=algo_psp, prob=prob_psp, pop_size=10)

    # 4 - Evolve and wait for the solution 
    archi.evolve()
    archi.wait()

    # 5 - Print best encoded actions and fitness obtained
    print(archi.get_champions_f())
    print(archi.get_champions_x())

if __name__ == "__main__":
    main()
