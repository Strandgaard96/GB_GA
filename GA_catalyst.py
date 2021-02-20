from rdkit import Chem
import numpy as np
import time
import crossover as co
import scoring_functions as sc
import GB_GA as ga 
import sys
from multiprocessing import Pool
import random
# from catalyst.scoring import rel_energy_scoring
from catalyst.relative_scoring import relative_scoring
from sa import reweigh_scores_by_sa, neutralize_molecules


seed=101
random.seed(seed)
np.random.seed(seed)

scoring_function = relative_scoring
n_confs = 15 # None calculates how many conformers based on 5+5*n_rot
scoring_args = n_confs

population_size = 4
mating_pool_size = 4
generations = 50
mutation_rate = 0.5
co.average_size = 25. 
co.size_stdev = 5.
prune_population = True
n_tries = 1
n_cpus = 4
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

def GA(args):
    population_size, file_name, scoring_function, generations, mating_pool_size, mutation_rate,\
    scoring_args, prune_population, n_cpus, sa_screening, seed = args

    np.random.seed(seed)
    random.seed(seed)

    population = ga.make_initial_population(population_size, file_name)
    prescores = sc.calculate_scores_parallel(population, scoring_function, scoring_args, n_cpus)
    # scores = normalize(prescores)
    scores = prescores

    if sa_screening:
        scores, sascores = reweigh_scores_by_sa(neutralize_molecules(population), scores)

    fitness = ga.calculate_normalized_fitness(scores)
    
    if sa_screening:
        print(f'{list(zip(scores, prescores, sascores, [Chem.MolToSmiles(mol) for mol in population]))}')
    else:
        print(f'{list(zip(scores, [Chem.MolToSmiles(mol) for mol in population]))}')

    high_scores = []
    for generation in range(generations):
        mating_pool = ga.make_mating_pool(population, fitness, mating_pool_size)
        new_population = ga.reproduce(mating_pool, population_size, mutation_rate)
        new_prescores = sc.calculate_scores_parallel(new_population, scoring_function, scoring_args, n_cpus)
            
        if sa_screening:
            new_scores, new_sascores = reweigh_scores_by_sa(neutralize_molecules(new_population), new_prescores)
            population, scores, prescores, sascores = ga.sanitize(population+new_population, scores+new_scores, population_size, prune_population, sa_screening, prescores+new_prescores, sascores+new_sascores)
        else:
            new_scores = new_prescores       
            population, scores = ga.sanitize(population+new_population, scores+new_scores, population_size, prune_population, sa_screening)
        
        fitness = ga.calculate_normalized_fitness(scores)

        high_scores.append((scores[0], Chem.MolToSmiles(population[0])))

        if sa_screening:
            print(f'{list(zip(scores, prescores, sascores, [Chem.MolToSmiles(mol) for mol in population]))}')
        else:
            print(f'{list(zip(scores, [Chem.MolToSmiles(mol) for mol in population]))}')

    return (scores, population, high_scores)

results = []
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
print(f'# Total duration: {(t1-t0)/60.0:.2f} minutes')