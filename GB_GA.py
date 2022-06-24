"""
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
"""

import numpy as np
import random

from rdkit import Chem
import crossover as co
import mutate as mu
from my_utils.my_utils import Individual, Generation
from scoring.make_structures import create_ligands, create_prim_amine
import copy


def read_file(file_name):
    """Read smiles from file and return mol list"""
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            mol_list.append(Chem.MolFromSmiles(smiles))

    return mol_list


def make_initial_population(population_size, file_name, rand=False):
    """

    Args:
        population_size (int): How many molecules in starting population
        file_name (str): Name of csv til to load molecules from
        rand (bool): Indicates whether molecules are randomly selected from file

    Returns:
        initial_population (Population)
    """
    mol_list = read_file(file_name)
    initial_population = Generation(generation_num=0)

    for i in range(population_size):
        if rand:

            # Check for any amines
            flag = False
            while not flag:
                mol = random.choice(mol_list)
                candidate_match = mol.GetSubstructMatches(
                    Chem.MolFromSmarts("[NX3;H2,H1,H0]")
                )
                if len(candidate_match) > 0:
                    flag = True

            # Check for prim amine
            match = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]"))
            if len(match) == 0:
                print(f"There are no primary amines to cut so creating new")
                ligand, cut_idx = create_prim_amine(mol)
                # If we cannot split, simply add methyl as ligand
                if not cut_idx:
                    ligand = Chem.MolFromSmiles("CN")
                    cut_idx = [[1]]
                initial_population.molecules.append(
                    Individual(ligand, cut_idx=cut_idx[0][0], original_mol=mol)
                )
            else:
                initial_population.molecules.append(
                    Individual(mol, cut_idx=random.choice(match)[0], original_mol=mol)
                )
        else:
            mol = mol_list[i]
            # Check for primary amine first
            match = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]"))
            if len(match) == 0:
                ligand, cut_idx = create_prim_amine(mol)
                initial_population.molecules.append(
                    Individual(ligand, cut_idx=cut_idx), original_mol=mol
                )
            else:
                initial_population.molecules.append(
                    Individual(mol, cut_idx=random.choice(match), original_mol=mol)
                )
    initial_population.generation_num = 0
    initial_population.assign_idx()
    return initial_population


def make_initial_population_debug(population_size, file_name, rand=False):
    """Function that runs localy and creates a small pop for debugging"""
    mol_list = read_file("data/ZINC_1000_amines.smi")
    initial_population = Generation(generation_num=0)

    smiles = ["CN", "CCN", "CCN", "CCN"]
    idx = [1, 2, 3, 4]

    for i in range(population_size):

        ligand = Chem.MolFromSmiles(smiles[i])
        cut_idx = [[idx[i]]]
        initial_population.molecules.append(Individual(ligand, cut_idx=cut_idx[0][0]))
    initial_population.generation_num = 0
    initial_population.assign_idx()
    return initial_population


# TODO convert to class method
def calculate_normalized_fitness(population):
    """
    Args:
        population (Population):

    Returns:
        None
    """

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
    """Select candidates from population based on fitness(score)"""
    fitness = population.get("normalized_fitness")
    mating_pool = []
    for _ in range(mating_pool_size):
        mating_pool.append(
            copy.deepcopy(np.random.choice(population.molecules, p=fitness))
        )

    return mating_pool  # list of Individuals


def reproduce(mating_pool, population_size, mutation_rate, molecule_filter):
    """Perform crossover operations on the mating pool"""
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
    return Generation(molecules=new_population)


def sanitize(molecules, population_size, prune_population):
    """Create a new population from the proposed molecules.
    If any molecules from newly scored molecules exists in population,
    we only select one. Finaly the prune class method is called to
    return only the top scoring molecules.

        molecules List(Individual)

    """
    if prune_population:
        smiles_list = []
        new_population = Generation()
        for individual in molecules:
            copy_individual = copy.deepcopy(individual)
            if copy_individual.smiles not in smiles_list:
                smiles_list.append(copy_individual.smiles)
                new_population.molecules.append(copy_individual)
    else:
        copy_population = copy.deepcopy(molecules)
        new_population = Generation(molecules=copy_population.molecules)

    new_population.prune(population_size)

    return new_population


def debug():

    mol = Chem.MolFromSmiles("c1ccc([C@@H]2C[C@H]2C[NH2+]Cc2nc3c(s2)CCC3)cc1")

    ligand, cut_idx = create_prim_amine(mol)


if __name__ == "__main__":
    debug()
