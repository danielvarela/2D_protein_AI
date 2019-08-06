import random

import gym
from gym import spaces
import numpy as np
from threading import Thread
from visual_utils.bokeh_visualizer import Visualizer
from cost_funcs.psp_func import PSPFunction
from cost_funcs.nn_folding_func import NeuralNetworkFunction 


def ensure_bounds(vec, bounds):

    vec_new = []
    # cycle through each variable in vector 
    for i, in enumerate(vec):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])
        
    return vec_new


class Individual():
    def __init__(self, genotype, score):
        self.genotype = genotype
        self.score = score
        
#--- MAIN ---------------------------------------------------------------------+

class DifferentialEvolutionAlgorithm():
    def __init__(self, seq, cost_func, popsize, mutate, recombination, maxiter, strategy =
                 "nn_operator" , visualizer=None):
        #func = NeuralNetworkFunction(seq)
        #cost_func = func                   # Cost function
        self.seq = seq
        self.cost_func = cost_func
        #self.cost_func = cost_func
        self.popsize = popsize
        self.mutate = mutate
        self.recombination = recombination
        self.maxiter = maxiter
        self.visualizer = visualizer
        self.strategy = strategy
        
    def main(self):
        #--- INITIALIZE A POPULATION (step #1) ----------------+
        #self.cost_func = PSPFunction(self.seq)
        ind_size = self.cost_func.size()
        self.bounds = [(-1,1)] * ind_size            # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
        population = []
        for i in range(0,self.popsize):
            indv = []
            for j in range(len(self.bounds)):
                indv.append(random.uniform(self.bounds[j][0],self.bounds[j][1]))
            indv_score = self.cost_func.run(indv)       
            build_ind = Individual(indv, indv_score)
            population.append(build_ind)
               
        #--- SOLVE --------------------------------------------+

        # cycle through each generation (step #2)
        for i in range(1,self.maxiter+1):
            print('GENERATION:' + str(i))

            gen_scores = [] # score keeping

            # cycle through each individual in the population
            for j in range(0, self.popsize):

                #--- MUTATION (step #3.A) ---------------------+

                # select three random vector index positions [0, self.popsize), not including current vector (j)
                candidates = list(range(0,self.popsize))
                candidates.remove(j)
                random_index = random.sample(candidates, 3)

                x_1 = population[random_index[0]].genotype
                x_2 = population[random_index[1]].genotype
                x_3 = population[random_index[2]].genotype
                x_t = population[j].genotype     # target individual

                # subtract x3 from x2, and create a new vector (x_diff)
                x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

                # multiply x_diff by the mutation factor (F) and add to x_1
                v_donor = [x_1_i + self.mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
                v_donor = ensure_bounds(v_donor, self.bounds)

                #--- RECOMBINATION (step #3.B) ----------------+

                v_trial = []
                for k, in enumerate(x_t):
                    crossover = random.random()
                    if crossover <= self.recombination:
                        v_trial.append(v_donor[k])

                    else:
                        v_trial.append(x_t[k])

                #--- GREEDY SELECTION (step #3.C) -------------+

                score_trial  = self.cost_func.run(v_trial)
                score_target = population[j].score

                if score_trial < score_target:
                    population[j] = Individual(v_trial, score_trial)
                    gen_scores.append(score_trial)
                else:
                    gen_scores.append(score_target)

            #--- SCORE KEEPING --------------------------------+

            gen_avg = sum(gen_scores) / self.popsize                         # current generation avg. fitness
            gen_best = min(gen_scores)                                  # fitness of best individual
            gen_sol = population[gen_scores.index(min(gen_scores))].genotype     # solution of best individual

            print('      > GENERATION AVERAGE: %f ' % gen_avg )
            print('      > GENERATION BEST: %f ' % gen_best )

            #self.visualizer.render_gen(i, population)
            self.cost_func.render(gen_sol)        

        return gen_sol

#--- MAIN  ----------------------------------------------------------------+

def build_cost_func(mode, seq, strategy):
    if (mode == "PSP"):
        return PSPFunction(seq)
    if (mode == "nn_folding"):
        return NeuralNetworkFunction(seq, strategy)
    
def main():
    #seq = 'HHHHHPHHHHHHPHHHHPHH' # Our input sequence
    #seq = 'HPHPPHHPHPPHPHHPPHPH' # Our input sequence
    seq = 'HPHPHPHPPHPH' # Our input sequence
    #func = CostFunction()
    #ind_size = func.size()
    #print(ind_size)
    #bounds = [(-1,1)] * ind_size           # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
    #popsize = len(seq) * 15            # Population size, must be >= 4
    popsize = 100                   # Population size, must be >= 4
    mutate = 0.3                  # Mutation factor [0,2]
    recombination = 0.9          # Recombination rate [0,1]
    maxiter = 5000                        # Max number of generations (maxiter)

    #visualizer = Visualizer(0 ,0)
    visualizer = None
    strategy  = "nn_operator_ext"
    mode = "PSP"
    cost_func = build_cost_func(mode, seq, strategy)

    #--- RUN ----------------------------------------------------------------------+
    alg = DifferentialEvolutionAlgorithm(seq, cost_func, popsize, mutate, recombination,
                                         maxiter, strategy, visualizer)
    alg.main()
    #thread = Thread(target=alg.main)
    #thread.start()


if __name__== "__main__":
  main()


#--- END ----------------------------------------------------------------------+
