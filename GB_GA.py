"""
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
"""

from rdkit import Chem

# from rdkit import rdBase
# rdBase.DisableLog('rdApp.error')

import numpy as np
import random

import crossover as co
import mutate as mu
from my_utils.my_utils import Individual, Population
import copy


def read_file(file_name):
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            mol_list.append(Chem.MolFromSmiles(smiles))

    return mol_list


from scoring.make_structures import create_ligands, create_prim_amine


def make_initial_population(population_size, file_name, rand=False):
    mol_list = read_file(file_name)
    initial_population = Population()

    for i in range(population_size):
        if rand:
            ligand = create_ligands(random.choice(mol_list))
            initial_population.molecules.append(Individual(ligand))
        else:
            ligand = create_ligands(mol_list[i])
            initial_population.molecules.append(Individual(ligand))

    initial_population.generation_num = 0
    initial_population.assign_idx()

    return initial_population


def make_initial_population_res(population_size, file_name, rand=False):
    mol_list = read_file(file_name)
    initial_population = Population()

    for i in range(population_size):
        if rand:
            # Check for primary amine
            mol = random.choice(mol_list)
            match = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2;!+1]"))
            if len(match) == 0:
                print(f"There are no primary amines to cut so creating new")
                ligand, cut_idx = create_prim_amine(random.choice(mol_list))
                # If we cannot split, simply add methyl as ligand
                if not cut_idx:
                    ligand = Chem.MolFromSmiles("CN")
                    cut_idx = [[0]]
                initial_population.molecules.append(
                    Individual(ligand, cut_idx=cut_idx[0][0])
                )
            else:
                initial_population.molecules.append(
                    Individual(mol, cut_idx=random.choice(match)[0])
                )
        else:
            mol = mol_list[i]
            # Check for primary amine firstlio
            match = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2;!+1]"))
            if len(match) == 0:
                ligand, cut_idx = create_prim_amine(random.choice(mol_list))
                initial_population.molecules.append(Individual(ligand, cut_idx=cut_idx))
            else:
                initial_population.molecules.append(
                    Individual(mol, cut_idx=random.choice(match))
                )
    initial_population.generation_num = 0
    initial_population.assign_idx()
    return initial_population


def calculate_normalized_fitness(population):

    # onvert to high and low scores.
    scores = population.get("score")
    scores = [-s for s in scores]

    min_score = np.nanmin(scores)
    shifted_scores = [0 if np.isnan(score) else score - min_score for score in scores]
    sum_scores = sum(shifted_scores)
    if sum_scores == 0:
        print(
            "WARNING: Shifted scores are zero. Normalized fitness is therefore dividing with "
            "zero, could be because the population only contains one individual"
        )

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


def reproduce_old(mating_pool, population_size, mutation_rate, filter):  # + filter
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


def reproduce(mating_pool, population_size, mutation_rate, molecule_filter):
    new_population = []
    counter = 0
    while len(new_population) < population_size:
        if random.random() > mutation_rate:
            parent_A = copy.deepcopy(random.choice(mating_pool))
            parent_B = copy.deepcopy(random.choice(mating_pool))
            new_child = co.crossover(
                parent_A.rdkit_mol, parent_B.rdkit_mol, molecule_filter
            )
            if new_child != None:
                new_child = Individual(rdkit_mol=new_child)
                new_population.append(new_child)
        else:
            parent = copy.deepcopy(random.choice(mating_pool))
            mutated_child, mutated = mu.mutate(parent.rdkit_mol, 1, molecule_filter)
            if mutated_child != None:
                mutated_child = Individual(
                    rdkit_mol=mutated_child,
                )
                new_population.append(mutated_child)
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
