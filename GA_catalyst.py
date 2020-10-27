from rdkit import Chem
import numpy as np
import time
import crossover as co
import scoring_functions as sc
import GB_GA as ga 
import sys
from multiprocessing import Pool
import random
from catalyst import cat_scoring

seed=123
random.seed(seed)
np.random.seed(seed)

scoring_function = cat_scoring
n_confs = 5 # calculates how many conformers based on 5+5*n_rot
scoring_args = n_confs

population_size = 20
mating_pool_size = 20
generations = 50
mutation_rate = 0.05
co.average_size = 25. 
co.size_stdev = 5.
prune_population = True
n_tries = 1
n_cpus = 20
sa_screening = True
seeds = np.random.randint(100000, size=2*n_tries)

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
print('* SA screening', sa_screening)
print('* seeds seed', seed)
print('* seeds', ','.join(map(str, seeds)))
print('')

results = []
size = []
t0 = time.time()
all_scores = []
generations_list = []

index = slice(0,n_tries) if prune_population else slice(n_tries,2*n_tries)
temp_args = [[population_size, file_name,scoring_function,generations,mating_pool_size,
                  mutation_rate,scoring_args, prune_population, n_cpus, sa_screening] for i in range(n_tries)]
args = []
for x,y in zip(temp_args,seeds[index]):
    x.append(y)
    args.append(x)

output = []
for i in range(n_tries):
    output.append(ga.GA(args[i]))


for i in range(n_tries):     
    #(scores, population) = ga.GA([population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate,scoring_args,prune_population])
    (scores, population, high_scores, high_prescores) = output[i]
    all_scores.append(scores)
    print(f'# Run {i+1}: Highest Scorer: {scores[0]:.2f} {Chem.MolToSmiles(population[0])} \nBest Mol in each Generation: {high_scores}')
    results.append(scores[0])
    generations_list.append(high_scores)
    #size.append(Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())

t1 = time.time()
# print('')
print(f'# max score {max(results):.2f}, mean {np.array(results).mean():.2f} +/- {np.array(results).std():.2f}')
# print(f'mean generations {np.array(generations_list).mean():.2f} +/- {np.array(generations_list).std():.2f}')
print(f'# Total duration: {(t1-t0)/60.0:.2f} minutes')
#print(max(size),np.array(size).mean(),np.array(size).std())
# print(f'# Generation list: {generations_list}')
print(f'# High Scores: {high_scores}')
print(f'# High Prescores: {high_prescores}')
# print(f'All scores:{all_scores}')