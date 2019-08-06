# Artificial intelligence methods applied to protein folding with 2D lattice

My PhD research project focuses on the computational modeling of the protein folding process. The goal is to obtain  
the three-dimensional structure of the protein only with the amino acid  sequence information (*ab initio*) by using evolutionary computing, artificial life techniques and complex systems theory methods. Since the structure defines the function of a protein, this would allow a computational drug design. More information about the protein structure prediction problem can be found at [Wikipedia](https://en.wikipedia.org/wiki/Protein_structure_prediction). More information about my research can be found at my publications or, in case of more detailed information, please, feel free to contact me.

This project is a python implementation of the same methods using in my PhD thesis but with a simple 2D lattice and known python libraries.

### Dependencies

* bokeh==1.2.0
* ann_visualizer==2.5
* matplotlib==2.0.0
* gym==0.12.5
* tornado==6.0.2
* numpy==1.12.1
* six==1.11.0
* Keras==2.2.4
* pytest==5.0.1
* scikit_learn==0.21.3
* pygmo==2.10

### Installation
This package is only compatible with Python 3.4 and above. To install this package, please follow the instructions below:

* Install OpenAI Gym and its dependencies.
* Install the package itself:

```
git clone https://github.com/danielvarela/2D_protein_AI.git
cd 2D_protein_AI
pip install -e .
```

### Basic Usage

```python
def main():  
  seq = 'HPHPHPHPPHPH' # Our input sequence  
  popsize = 100 # Population size, must be >= 4   	
  mutate = 0.3 # Mutation factor [0,2]   
  recombination = 0.9 # crossover factor [0,1]
  maxiter = 5000 # Max number of generations (maxiter)
  cost_func = build_cost_func("PSP", seq)
  
  #--- RUN ------------------------------------------------------------------+
  alg = DifferentialEvolutionAlgorithm(seq, cost_func, popsize, mutate, recombination, maxiter)
  alg.main()

python DE.py
```


## Protein Structure Prediction

One of the most important problems in molecular biology is to obtain the native structure of a protein from its primary structure, i.e., the amino acids chain. Ab-initio methods adopt different approaches for the protein structure representation. For example, lattice models impose the constraint that the location of amino acids must be in the lattice sites. In the *ab initio* protein structure prediction problem (PSP) many authors have been working on the use of search methods, specially evolutionary algorithms, employing the simple HP lattice model.


## Differential Evolution Algorithm

Differential Evolution [Price05] is a population-based search method. DE creates new candidate solutions by combining existing ones according to a simple formula of vector crossover and mutation, and then keeping whichever candidate solution has the best score or fitness on the optimization problem at hand. 


### Moves

Each connection between two amino acids can perform four possible moves in a 2D lattice environment:  `0`  (left),  `1`  (down),  `2`  (up), and  `3`  (right). 

### Individuals

Each individual is encoded as an 1D array of length `n * {L, D, U, R}`, where n is the total number of connections between the amino acid sequence. By this way, we encode each sequence move as the minimum of its four possible moves.

![img](https://github.com/danielvarela/2D_protein_AI/blob/master/images/individual_encoding.png)

### Fitness

This implementation uses the one of the most studied lattice models, the HP model [Dill90]. Proteins that have minimum energy are assumed to be in their native state. Two amino acids are in contact if they are non-consecutive neighbors on the protein sequence and are neighbors (or in contact) on the lattice. In case that the two amino acids are hydrohopic (H), we count the contact.

![equation](https://latex.codecogs.com/gif.latex?E&space;=&space;\sum_{i&space;<&space;j&space;&space;1}&space;c_{ij}&space;\cdot&space;e_{ij})


### pygmo implementation

Pagmo (C++) or pygmo (Python) is a scientific library that provides an unified interface to optimization algorithms and to optimization problems and to make their deployment in massively parallel environments easy. More information can be found at its website [pygmo url](https://esa.github.io/pagmo2/index.html)

at */pygmo_psp/main.py*, an implementation using the pygmo library can be found. The main function is easy to follow and clear for run an small example using a pygmo archipelago with differential evolution algorithm.

## Artificial Neural Network to obtain the temporal folding process

We used cellular automata (CA) for the modeling of the temporal folding of proteins. Unlike the focus of the vast research already done on the direct prediction of the final folded conformations, we will model the temporal and dynamic folding process. The CA model defines how the amino acids interact through time to obtain a folded conformation. We employed the TIP model to represent the protein conformations in a lattice, we extended the classical CA models using artificial neural networks for their implementation, and we used evolutionary computing to automatically obtain the models by means of Differential Evolution. Moreover, the modeling of the folding provides the final protein conformation.

### Basic Usage

To use this algorithm, simply change this line at the main function of the DE.py document

```python
  cost_func = build_cost_func("PSP", seq)
```
by

```python
  strategy = "nn_operator"
  cost_func = build_cost_func("nn_folding", seq, strategy)
```

The strategy parameter selects different ways to applied the ANN to the protein moves. I would like to explain further this in the future.

### Individuals

We use Differential Evolution to obtain an optimized feed-forward artificial neural network. Each individual encodes the neural network weights as an 1D array.

### Fitness

To evaluate each individual, the feed-forward artificial neural network is applied iteratively to each connection between amino acids in order to obtain the next move. Once the neural network finnish this process, the completed folded protein is evaluated using the HP fitness function previously explained. By this way, the fitness of the candidate neural network corresponds to the folded protein that it can obtain.

![ann_folding](https://github.com/danielvarela/2D_protein_AI/blob/master/images/figure_ann.png)

