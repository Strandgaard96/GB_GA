from rdkit import Chem
import numpy as np
import time
from multiprocessing import Pool
import random
import logging

import sys
sys.path.append('/home/julius/soft/GB-GA/')

import crossover as co
import scoring_functions as sc
import GB_GA as ga 
from sa import reweigh_scores_by_sa, neutralize_molecules
from catalyst.ts_scoring import ts_scoring

randomseed=101
random.seed(randomseed)
np.random.seed(randomseed)

timing_formatter = logging.Formatter('%(message)s')
timing_logger = logging.getLogger('timing')
timing_logger.setLevel(logging.INFO)
timing_file_handler = logging.FileHandler('scoring_timings.log')
timing_file_handler.setFormatter(timing_formatter)
timing_logger.addHandler(timing_file_handler)

warning_formatter = logging.Formatter('%(levelname)s:%(message)s')
warning_logger = logging.getLogger('warning')
warning_logger.setLevel(logging.WARNING)
warning_file_handler = logging.FileHandler('scoring_warning.log')
warning_file_handler.setFormatter(warning_formatter)
warning_logger.addHandler(warning_file_handler)

directory = sys.argv[-1]
file_name = sys.argv[-2]

scoring_function = ts_scoring
n_confs = 1 # None calculates how many conformers based on 5+5*n_rot
scoring_args = [n_confs, randomseed, timing_logger, warning_logger, directory]

population_size = 8
mating_pool_size = 8
generations = 3
mutation_rate = 0.5
co.average_size = 25. 
co.size_stdev = 5.
prune_population = True
n_tries = 1
n_cpus = 8   # this has to be same as in submit script
sa_screening = True
seeds = np.random.randint(100000, size=2*n_tries)

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
print('* seeds seed', randomseed)
print('* seeds', ','.join(map(str, seeds)))
print('')

def GA(args):
    population_size, file_name, scoring_function, generations, mating_pool_size, mutation_rate,\
    scoring_args, prune_population, n_cpus, sa_screening, randomseed = args

    np.random.seed(randomseed)
    random.seed(randomseed)
    generation = 0
    population = ga.make_initial_population(population_size, file_name)
    prescores = sc.calculate_scores_parallel(population, scoring_function, scoring_args, n_cpus, generation) # Energy of TS, the smaller the better
    # scores = normalize(prescores)
    inv_scores = [element * -1 for element in prescores] # The Larger the Better

    if sa_screening:
        scores, sascores = reweigh_scores_by_sa(neutralize_molecules(population), inv_scores)
    else:
        scores = inv_scores

    fitness = ga.calculate_normalized_fitness(scores)
    
    if sa_screening:
        print(f'{list(zip(scores, prescores, sascores, [Chem.MolToSmiles(mol) for mol in population]))}')
    else:
        print(f'{list(zip(scores, [Chem.MolToSmiles(mol) for mol in population]))}')

    high_scores = []
    for generation in range(generations):
        mating_pool = ga.make_mating_pool(population, fitness, mating_pool_size)
        new_population = ga.reproduce(mating_pool, population_size, mutation_rate)
        new_prescores = sc.calculate_scores_parallel(new_population, scoring_function, scoring_args, n_cpus, generation+1)
        new_inv_score = [element * -1 for element in new_prescores]

        if sa_screening:
            new_scores, new_sascores = reweigh_scores_by_sa(neutralize_molecules(new_population), new_inv_score)
            population, scores, prescores, sascores = ga.sanitize(population+new_population, scores+new_scores, population_size, prune_population, sa_screening, prescores+new_prescores, sascores+new_sascores)
        else:
            new_scores = new_inv_score       
            population, scores = ga.sanitize(population+new_population, scores+new_scores, population_size, prune_population, sa_screening)
        
        fitness = ga.calculate_normalized_fitness(scores)

        high_scores.append((scores[0], Chem.MolToSmiles(population[0])))

        if sa_screening:
            print(f'{list(zip(scores, prescores, sascores, [Chem.MolToSmiles(mol) for mol in population]))}')
        else:
            print(f'{list(zip(scores, [Chem.MolToSmiles(mol) for mol in population]))}')

    return (scores, population, high_scores)

# Get Timings
timing_logger.info(f'# Running on {n_cpus} cores')

results = []
t0 = time.time()
all_scores = []
generations_list = []

index = slice(0,n_tries) if prune_population else slice(n_tries,2*n_tries)
temp_args = [[population_size, file_name, scoring_function, generations, mating_pool_size,
                  mutation_rate, scoring_args, prune_population, n_cpus, sa_screening] for i in range(n_tries)]
args = []
for x,y in zip(temp_args,seeds[index]):
    x.append(y)
    args.append(x)

# Run the GA
output = []
for i in range(n_tries):
    output.append(GA(args[i]))


for i in range(n_tries):     
    (scores, population, high_scores) = output[i]
    all_scores.append(scores)
    results.append(scores[0])
    generations_list.append(high_scores)

t1 = time.time()
print(f'* Total duration: {(t1-t0)/60.0:.2f} minutes')