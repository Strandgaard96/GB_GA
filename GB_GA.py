"""
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

# from rdkit import rdBase
# rdBase.DisableLog('rdApp.error')

import numpy as np
import random
import time
import sys

import crossover as co
import mutate as mu
import scoring_functions as sc
from sa import reweigh_scores_by_sa, neutralize_molecules
from catalyst.utils import Population, Individual
import copy


def read_file(file_name):
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            mol_list.append(Chem.MolFromSmiles(smiles))

    return mol_list


def make_initial_population(population_size, file_name, rand=False):
    mol_list = read_file(file_name)
    initial_population = Population()
    for i in range(population_size):
        if rand:
            initial_population.molecules.append(Individual(random.choice(mol_list)))
        else:
            initial_population.molecules.append(Individual(mol_list[i]))
    initial_population.generation_num = 0
    initial_population.assign_idx()

    return initial_population


def calculate_normalized_fitness(population):
    scores = population.get("score")
    min_score = np.min(scores)
    shifted_scores = [score - min_score for score in scores]
    sum_scores = sum(shifted_scores)
    for individual, shifted_score in zip(population.molecules, shifted_scores):
        individual.normalized_fitness = shifted_score / sum_scores


def make_mating_pool(population, mating_pool_size):
    fitness = population.get("normalized_fitness")
    mating_pool = []
    for _ in range(mating_pool_size):
        mating_pool.append(
            copy.deepcopy(np.random.choice(population.molecules, p=fitness))
        )

    return mating_pool  # list of Individuals


def reproduce(mating_pool, population_size, mutation_rate, filter):  # + filter
    """Creates a new population based on the mating_pool"""
    new_population = []
    while len(new_population) < population_size:
        parent_A = copy.deepcopy(random.choice(mating_pool))
        parent_B = copy.deepcopy(random.choice(mating_pool))
        new_child = co.crossover(parent_A.rdkit_mol, parent_B.rdkit_mol, filter)
        if new_child != None:
            mutated_child, mutated = mu.mutate(new_child, mutation_rate, filter)
            if mutated_child != None:
                # print(','.join([Chem.MolToSmiles(mutated_child),Chem.MolToSmiles(new_child),Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B)]))
                new_population.append(
                    Individual(
                        rdkit_mol=mutated_child,
                        parentA_idx=parent_A.idx,
                        parentB_idx=parent_B.idx,
                        mutated=mutated,
                    )
                )

    return Population(molecules=new_population)


def reproduce2(
    mating_pool, population_size, mutation_rate=0.5, crossover_rate=1, filter=None
):
    new_population = []
    succeeded = True
    counter = 0
    while len(new_population) < population_size and counter < 10:
        parent_A = copy.deepcopy(random.choice(mating_pool))
        parent_B = copy.deepcopy(random.choice(mating_pool))
        parent_A_idx = parent_A.idx
        parent_B_idx = parent_B.idx
        if succeeded:
            x = random.random()
        if x < crossover_rate:
            new_child = co.crossover(parent_A.rdkit_mol, parent_B.rdkit_mol, filter)
            if new_child == None:
                succeeded = False
                counter += 1
                continue
        else:
            new_child = parent_A.rdkit_mol
            parent_A_idx = None
            parent_B_idx = None

        mutated_child, mutated = mu.mutate(new_child, mutation_rate, filter)
        if mutated_child == None:
            succeeded = False
            counter += 1
            continue

        new_population.append(
            Individual(
                rdkit_mol=mutated_child,
                parentA_idx=parent_A_idx,
                parentB_idx=parent_B_idx,
                mutated=mutated,
            )
        )
        succeeded = True
        counter = 0

    return Population(molecules=new_population)


def sanitize(molecules, population_size, prune_population):
    if prune_population:
        smiles_list = []
        new_population = Population()
        for individual in molecules:
            copy_individual = copy.deepcopy(individual)
            if copy_individual.smiles not in smiles_list:
                smiles_list.append(copy_individual.smiles)
                new_population.molecules.append(copy_individual)
    else:
        copy_population = copy.deepcopy(molecules)
        new_population = Population(molecules=copy_population.molecules)

    new_population.prune(population_size)

    return new_population


if __name__ == "__main__":
    pass
