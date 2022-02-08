'''
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
'''

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')

import numpy as np
import random
import heapq
import time
import sys

from scipy.stats import rankdata

import crossover as co
import mutate as mu
import scoring_functions as sc
from catalyst.utils import Individual

def read_file(file_name):
  mol_list = []
  with open(file_name,'r') as file:
    for smiles in file:
      mol_list.append(Chem.MolFromSmiles(smiles))

  return mol_list

def make_initial_population(population_size, file_name, random=True):
    if random:
        with open(file_name) as fin:
            sample = heapq.nlargest(population_size, fin, key=lambda L: random.random())
    else:
        with open(file_name) as fin:
            sample = [smiles for smiles in fin][:population_size]
    population = [Chem.MolFromSmiles(smi.rstrip()) for smi in sample]

    return population

def calculate_normalized_fitness(scores):
  sum_scores = sum(scores)
  normalized_fitness = [score/sum_scores for score in scores]

  return normalized_fitness

def calculate_fitness(
    scores, minimization=False, selection="roulette", selection_pressure=None
):
    if minimization:
        scores = [-s for s in scores]
    if selection == "roulette":
        fitness = scores
    elif selection == "rank":
        scores = [
            float("-inf") if np.isnan(x) else x for x in scores
        ]  # works for minimization
        ranks = rankdata(scores, method="ordinal")
        n = len(ranks)
        if selection_pressure:
            fitness = [
                2
                - selection_pressure
                + (2 * (selection_pressure - 1) * (rank - 1) / (n - 1))
                for rank in ranks
            ]
        else:
            fitness = [r / n for r in ranks]
    else:
        raise ValueError(
            f"Rank-based ('rank') or roulette ('roulette') selection are available, you chose {selection}."
        )

    return fitness

def make_mating_pool(population, fitness, mating_pool_size):
    mating_pool = []
    for i in range(mating_pool_size):
        mating_pool.append(random.choices(population, weights=fitness, k=1)[0])
    return mating_pool
 

def reproduce(mating_pool, population_size, mutation_rate, molecule_filter, generation):
    new_population = []
    counter = 0
    while len(new_population) < population_size:
        if random.random() > mutation_rate:
            parent_A = random.choice(mating_pool)
            parent_B = random.choice(mating_pool)
            new_child = co.crossover(
                parent_A.rdkit_mol, parent_B.rdkit_mol, molecule_filter
            )
            if new_child != None:
                idx = (generation, counter)
                counter += 1
                new_child = Individual(rdkit_mol=new_child, idx=idx)
                new_population.append(new_child)
        else:
            parent = random.choice(mating_pool)
            mutated_child = mu.mutate(parent.rdkit_mol, 1, molecule_filter)
            if mutated_child != None:
                idx = (generation, counter)
                counter += 1
                mutated_child = Individual(
                    rdkit_mol=mutated_child,
                    idx=idx,
                )
                new_population.append(mutated_child)
    return new_population

def sanitize(population, population_size, prune_population):
    if prune_population:
        sanitized_population = []
        for ind in population:
            if ind.smiles not in [si.smiles for si in sanitized_population]:
                sanitized_population.append(ind)
    else:
        sanitized_population = population

    sanitized_population.sort(
        key=lambda x: float("inf") if np.isnan(x.score) else x.score
    )  # np.nan is highest value, works for minimization of score

    new_population = sanitized_population[:population_size]
    return new_population  # selects individuals with lowest values
