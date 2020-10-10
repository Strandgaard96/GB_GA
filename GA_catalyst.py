from rdkit import Chem
import numpy as np
import time
import crossover as co
import scoring_functions as sc
import GB_GA as ga 
import sys
from multiprocessing import Pool
import random

random.seed(123)

scoring_function = sc.cat_scoring
n_confs = None # calculates how many conformers based on 5+5*n_rot
scoring_args = n_confs

population_size = 12
mating_pool_size = 12
generations = 10
mutation_rate = 0.05
co.average_size = 50. 
co.size_stdev = 5.
prune_population = False
n_tries = 1
n_cpus = 12
seeds = np.random.randint(100_000, size=2*n_tries)

file_name = sys.argv[1]


print('* n_confs', n_confs)
print('* population_size', population_size)
print('* mating_pool_size', mating_pool_size)
print('* generations', generations)
print('* mutation_rate', mutation_rate)
print('* average_size/size_stdev', co.average_size, co.size_stdev)
print('* initial pool', file_name)
print('* prune population', prune_population)
print('* number of tries', n_tries)
print('* number of CPUs', n_cpus)
print('* SMILES input file', file_name)
print('* seeds', ','.join(map(str, seeds)))
print('')

results = []
size = []
t0 = time.time()
all_scores = []
generations_list = []

index = slice(0,n_tries) if prune_population else slice(n_tries,2*n_tries)
temp_args = [[population_size, file_name,scoring_function,generations,mating_pool_size,
                  mutation_rate,scoring_args, prune_population, n_cpus] for i in range(n_tries)]
args = []
for x,y in zip(temp_args,seeds[index]):
    x.append(y)
    args.append(x)

output = []
for i in range(n_tries):
    output.append(ga.GA(args[i]))


for i in range(n_tries):     
    #(scores, population) = ga.GA([population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate,scoring_args,prune_population])
    (scores, population, generation) = output[i]
    all_scores.append(scores)
    print(f'{i} {scores[0]:.2f} {Chem.MolToSmiles(population[0])} {generation}')
    results.append(scores[0])
    generations_list.append(generation)
    #size.append(Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())

t1 = time.time()
# print('')
print(f'max score {max(results):.2f}, mean {np.array(results).mean():.2f} +/- {np.array(results).std():.2f}')
print(f'mean generations {np.array(generations_list).mean():.2f} +/- {np.array(generations_list).std():.2f}')
print(f'Total duration: {(t1-t0)/60.0:.2f} minutes')
#print(max(size),np.array(size).mean(),np.array(size).std())
print(generations_list)
print(all_scores)