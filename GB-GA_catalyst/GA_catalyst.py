# %%
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
from catalyst.utils import Generation


randomseed=9
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
n_cpus = int(sys.argv[-2])

# directory = '.'
# n_cpus = 4

scoring_function = ts_scoring
n_confs = 1 # None calculates how many conformers based on 5+5*n_rot
scoring_args = [n_confs, randomseed, timing_logger, warning_logger, directory]

file_name = '/home/julius/soft/GB-GA/ZINC_1000_amines.smi'
population_size = 15
mating_pool_size = 15
generations = 15
mutation_rate = 0.5
co.average_size = 25. 
co.size_stdev = 5.
prune_population = True
n_tries = 1
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
# %%

def GA(args):
    population_size, file_name, scoring_function, generations, mating_pool_size, mutation_rate,\
    scoring_args, prune_population, n_cpus, sa_screening, randomseed = args

    np.random.seed(randomseed)
    random.seed(randomseed)
    population = ga.make_initial_population(population_size, file_name)
    sc.calculate_scores_parallel(population=population, function=scoring_function, scoring_args=scoring_args, n_cpus=n_cpus)
    
    if sa_screening:
        neutralize_molecules(population)
        reweigh_scores_by_sa(population)
    
    population.sortby('score')
    gen = Generation(generation_num=0, children=population, survivors=population)
    gen.save('.')
    gen.print()

    for generation in range(generations):
        generation_num = generation+1
        # Making new Children
        ga.calculate_normalized_fitness(population)
        mating_pool = ga.make_mating_pool(population, mating_pool_size)
        new_population = ga.reproduce(mating_pool, population_size, mutation_rate, filter=None)
        new_population.generation_num = generation_num
        new_population.assign_idx()

        sc.calculate_scores_parallel(population=new_population, function=scoring_function, scoring_args=scoring_args, n_cpus=n_cpus)
      
        if sa_screening:
            neutralize_molecules(new_population)
            reweigh_scores_by_sa(new_population)
        new_population.sortby('score')

        # Select best Individuals from old and new population
        population = ga.sanitize(population.molecules+new_population.molecules, population_size, prune_population) # SURVIVORS
        population.generation_num = generation_num
        population.assign_idx()

        gen = Generation(generation_num=generation_num, children=new_population, survivors=population)
        gen.save('.')
        gen.print()


# %%

# Get Timings
timing_logger.info(f'# Running on {n_cpus} cores')


t0 = time.time()

args = [population_size, file_name, scoring_function, generations, mating_pool_size,
                  mutation_rate, scoring_args, prune_population, n_cpus, sa_screening, randomseed]
# Run the GA
GA(args)

t1 = time.time()
print(f'\n* Total duration: {(t1-t0)/60.0:.2f} minutes')

