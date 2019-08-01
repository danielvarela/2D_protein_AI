# Artificial intelligence methods applied to protein folding with 2D lattice

My PhD research project focuses on the computational modeling of the protein folding process. The goal is to obtain  
the three-dimensional structure of the protein only with the amino acid  sequence information (*ab initio*) by using evolutionary computing, artificial life techniques and complex systems theory methods. Since the structure defines the function of a protein, this would allow a computational drug design. More information about the protein structure prediction problem can be found at [Wikipedia](https://en.wikipedia.org/wiki/Protein_structure_prediction). To know more about my research can be found at my publications or, in case of more detailed information, please, feel free to contact me.

This project is a python implementation of the same methods using in my PhD thesis but with a simple 2D lattice and known python libraries.

## Protein Structure Prediction

One of the most important problems in molecular biology is to obtain the native structure of a protein from its primary structure, i.e., the amino acids chain. Ab-initio methods adopt different approaches for the protein structure representation. For example, lattice models impose the constraint that the location of amino acids must be in the lattice sites. In the *ab initio* protein structure prediction problem (PSP) many authors have been working on the use of search methods, specially evolutionary algorithms, employing the simple HP lattice model.


## Differential Evolution Algorithm

Differential Evolution [Price05] is a population-based search method. DE creates new candidate solutions by combining existing ones according to a simple formula of vector crossover and mutation, and then keeping whichever candidate solution has the best score or fitness on the optimization problem at hand. 


### Basic Usage

    def main():  
	   seq = 'HPHPHPHPPHPH' # Our input sequence  
	   popsize = 100 # Population size, must be >= 4   	
	   mutate = 0.3 # Mutation factor [0,2]   
	   recombination = 0.9  
	   maxiter = 5000 # Max number of generations (maxiter)   
	   alg = DifferentialEvolutionAlgorithm(seq, popsize, mutate, recombination,   maxiter)   
	   alg.main()

    python DE.py

### Moves

Each connection between two amino acids can perform four possible moves in a 2D lattice environment:  `0`  (left),  `1`  (down),  `2`  (up), and  `3`  (right). 

### Individuals

Each individual is encoded as an 1D array of $n * {L, D, U, R}$, where n is the total number of connections between the amino acid sequence. By this way, we encode each sequence move as the minimum of its four possible moves.

### Fitness

This implementation uses the one of the most studied lattice models, the HP model [Dill90]. Proteins that have minimum energy are assumed to be in their native state. The energy of a protein conformation is defined as:  
 
 $E = \sum_{i < j - 1} c_{ij} \cdot e_{ij}$ 
  
where $c_{ij}=1$ if amino acids $i$ and $j$ are non-consecutive neighbors on the protein sequence and are neighbors (or in contact) on the lattice, otherwise $0$; The term $e_{ij}$ depends on  the type of amino acids: $e_{ij}=-1$ if $ith$ and $jth$ amino acids are hydrophobic (H), otherwise $0$.


