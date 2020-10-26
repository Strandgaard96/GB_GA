'''
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
'''

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import numpy as np
import random
import time
import sys

import crossover as co
import mutate as mu
import scoring_functions as sc
from sa import reweigh_scores_by_sa, neutralize_molecules

def read_file(file_name):
  mol_list = []
  with open(file_name,'r') as file:
    for smiles in file:
      mol_list.append(Chem.MolFromSmiles(smiles))

  return mol_list

def make_initial_population(population_size,file_name):
  mol_list = read_file(file_name)
  population = []
  for i in range(population_size):
    population.append(random.choice(mol_list))
    
  return population

def calculate_normalized_fitness(scores): 
  min_score = np.min(scores)
  shifted_scores = [score-min_score for score in scores]
  sum_scores = sum(shifted_scores) 
  normalized_fitness = [score/sum_scores for score in shifted_scores]
  return normalized_fitness

def make_mating_pool(population,fitness,mating_pool_size):
  mating_pool = []
  for i in range(mating_pool_size):
  	mating_pool.append(np.random.choice(population, p=fitness))

  return mating_pool 


def reproduce(mating_pool,population_size,mutation_rate):
  new_population = []
  while len(new_population) < population_size:
    parent_A = random.choice(mating_pool)
    parent_B = random.choice(mating_pool)
    new_child = co.crossover(parent_A,parent_B)
    if new_child != None:
      mutated_child = mu.mutate(new_child,mutation_rate)
      if mutated_child != None:
        #print(','.join([Chem.MolToSmiles(mutated_child),Chem.MolToSmiles(new_child),Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B)]))
        new_population.append(mutated_child)

  return new_population

def sanitize(population,scores,prescores,population_size, prune_population):
    if prune_population:
      smiles_list = []
      population_tuples = []
      for score, prescore, mol in zip(scores,prescores,population):
          smiles = Chem.MolToSmiles(mol)
          if smiles not in smiles_list:
              smiles_list.append(smiles)
              population_tuples.append((score,prescore,mol))
    else:
      population_tuples = list(zip(scores,prescores,population))

    population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
    new_population = [t[2] for t in population_tuples]
    new_prescores = [t[1] for t in population_tuples]
    new_scores = [t[0] for t in population_tuples]

    return new_population, new_scores, new_prescores

def GA(args):
  population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate, \
  scoring_args , prune_population, n_cpus, sa_screening, seed = args

  np.random.seed(seed)
  random.seed(seed)

  high_scores = [] 
  high_prescores = []
  population = make_initial_population(population_size,file_name)
  print(f'# Initial Population')
  start = time.time()
  prescores = sc.calculate_scores_parallel(population,scoring_function,scoring_args,n_cpus)
  if sa_screening:
    scores = reweigh_scores_by_sa(neutralize_molecules(population), prescores)
  else:
    scores = prescores
  #reorder so best score comes first
  population, scores, prescores = sanitize(population, scores, prescores, population_size, False)  
  fitness = calculate_normalized_fitness(scores)
  high_scores.append((scores[0],Chem.MolToSmiles(population[0])))
  high_prescores.append((prescores[0],Chem.MolToSmiles(population[0])))
  print(f'{list(zip(scores, prescores, [Chem.MolToSmiles(mol) for mol in population]))}')

  for generation in range(generations):
    start = time.time()
    print(f'# Generation {generation+1}/{generations}')
    mating_pool = make_mating_pool(population,fitness,mating_pool_size)
    new_population = reproduce(mating_pool,population_size,mutation_rate)
    new_prescores = sc.calculate_scores_parallel(new_population,scoring_function,scoring_args, n_cpus)
    if sa_screening:
      new_scores = reweigh_scores_by_sa(neutralize_molecules(new_population), new_prescores)
      assert len(new_scores) == len(new_population)
    else:
      new_scores = new_prescores
    population, scores, prescores = sanitize(population+new_population, scores+new_scores, prescores+new_prescores, population_size, prune_population)  
    fitness = calculate_normalized_fitness(scores)
    high_scores.append((scores[0],Chem.MolToSmiles(population[0])))
    high_prescores.append((prescores[0],Chem.MolToSmiles(population[0])))
    print(f'{list(zip(scores, prescores, [Chem.MolToSmiles(mol) for mol in population]))}')
  return (scores, population, high_scores, high_prescores)


if __name__ == "__main__":
    pass
