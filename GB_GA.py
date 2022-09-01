"""
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
"""

import copy
import random

import numpy as np
from rdkit import Chem

import crossover as co
import mutate as mu
from my_utils.classes import Generation, Individual
from scoring.make_structures import create_prim_amine_revised


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
        initial_population Generation (class)
    """
    mol_list = read_file(file_name)
    initial_population = Generation(generation_num=0)

    for i in range(population_size):
        if rand:

            # Randomly coose mol until we find something with any amines
            candidate_match = None
            while not candidate_match:
                mol = random.choice(mol_list)
                candidate_match = mol.GetSubstructMatches(
                    Chem.MolFromSmarts("[NX3;H2,H1,H0,$(*n);!$(*N)]")
                )

            # Check for prim amine
            match = mol.GetSubstructMatches(
                Chem.MolFromSmarts("[NX3;H2;!$(*n);!$(*N)]")
            )
            if not match:
                print(f"There are no primary amines to cut so creating new")
                ligand, cut_idx = create_prim_amine_revised(mol)
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
            match = mol.GetSubstructMatches(
                Chem.MolFromSmarts("[NX3;H2;!$(*n);!$(*N)]")
            )
            if not match == 0:
                ligand, cut_idx = create_prim_amine_revised(mol)
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

    smiles = ["CCN", "CCCN", "CCN", "CCN"]
    idx = [2, 3, 2, 2]

    for i in range(population_size):

        ligand = Chem.MolFromSmiles(smiles[i])
        cut_idx = [[idx[i]]]
        initial_population.molecules.append(Individual(ligand, cut_idx=cut_idx[0][0]))
    initial_population.generation_num = 0
    initial_population.assign_idx()
    return initial_population


def make_mating_pool(population, mating_pool_size):
    """Select candidates from population based on fitness(score)

    Args:
        population Generation(class): The generation object
        mating_pool_size (int): The size of the mating pool

    Returns:
        mating_pool List(Individual): List of Individual objects
    """
    fitness = population.get("normalized_fitness")
    mating_pool = []
    for _ in range(mating_pool_size):
        mating_pool.append(
            copy.deepcopy(np.random.choice(population.molecules, p=fitness))
        )

    return mating_pool


def reproduce(mating_pool, population_size, mutation_rate, molecule_filter):
    """
    Perform crossover operating on the molecules in the mating pool

    Args:
        mating_pool List(Individual): List containing ind objects
        population_size (int): Size of whole population
        mutation_rate (float): Determines how often a
            mutation vs crossover operation is performed
        molecule_filter List(Chem.rdchem.Mol): List of smart pattern mol objects
            that ensure that toxic, etc molecules are not evolved

    Returns:
        Generation(class): The object holding the new population
    """
    new_population = []
    counter = 0
    while len(new_population) < population_size:
        if random.random() > mutation_rate:
            parent_A = copy.deepcopy(random.choice(mating_pool))
            parent_B = copy.deepcopy(random.choice(mating_pool))
            new_child = co.crossover(
                parent_A.rdkit_mol, parent_B.rdkit_mol, molecule_filter
            )
            if new_child:
                new_child = Individual(rdkit_mol=new_child)
                new_population.append(new_child)
        else:
            parent = copy.deepcopy(random.choice(mating_pool))
            mutated_child, mutated = mu.mutate(parent.rdkit_mol, 1, molecule_filter)
            if mutated_child:
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

    Args:
        molecules List(Individual): List of molecules to operate on.
            Contains newly scord molecules and the current best molecules.
        population_size (int): How many molecules allowed in population.
        prune_population (bool): Flag to select the top scoring molecules.

    Returns:

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
