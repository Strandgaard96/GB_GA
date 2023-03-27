"""Written by Jan H. and modified by Magnus Strandgaard 2023.

Jensen 2018.
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
"""

import logging
import math
import os
import pickle
import random
from typing import List

import numpy as np
from catalystGA import Ligand, Metal
from catalystGA.ga import GA
from rdkit import Chem

import crossover as co
import mutate as mu
from schrock import Schrock
from utils.classes import DataLoader, Individual, OutputHandler, ScoringFunction
from utils.utils import read_file

metals_list = [Metal("Mo")]

# Get mol generator from file
mol_generator = read_file("/home/magstr/Documents/GB_GA/data/ZINC_250k.smi")

ligands_list = []
for i, elem in enumerate(mol_generator):
    ligands_list.append(Ligand(elem))
    if i == 100:
        break


class GeneticAlgorithm(GA):
    """

    Args:
        args(dict): Dictionary containing all the commandline input args.

    Returns:
        gen: MoleculeHandler class that contains the results of the final generation
    """

    def __init__(self, args, mol_options):
        self.args = args
        self.generation_num = 0
        self.data_loader = DataLoader(args)
        self.scorer = ScoringFunction(args)
        self.output_handler = OutputHandler(args)
        super().__init__(mol_options)
        # self.sascorer = SaScorer()

    def reproduce(self, population_size, mutation_rate, molecule_filter) -> list:

        new_population = []
        # Run mutation and crossover until we have N = population_size
        while len(new_population) < population_size:
            if random.random() > mutation_rate:
                parent1, parent2 = np.random.choice(
                    self.molecules,
                    p=[ind.fitness for ind in self.molecules],
                    size=2,
                    replace=False,
                )
                new_child = co.crossover(
                    parent1.rdkit_mol, parent2.rdkit_mol, molecule_filter
                )
            else:
                mutate_parent = np.random.choice(
                    mating_pool,
                    p=[ind.fitness for ind in mating_pool],
                    size=1,
                    replace=False,
                )
                new_child, mutated = mu.mutate(
                    mutate_parent.rdkit_mol, 1, molecule_filter
                )

            if new_child:
                new_population.append(Individual(rdkit_mol=new_child))
        return new_population

    def prune(self, population: list) -> list:
        """Keep the best individuals in the population, cut down to
        'population_size'.

        Args:
            population (list): List of all individuals

        Returns:
            list: List of kept individuals
        """
        # TODO CHEECK THAT THIS SET WORK AS EXPECTED
        tmp = list(set(population))
        tmp.sort(
            key=lambda x: (self.maximize_score - 0.5) * float("-inf")
            if math.isnan(x.score)
            else x.score,
            reverse=False,
        )
        return tmp[: self.args["population_size"]]

    def reweigh_rotatable_bonds(self, nrb_target=4, nrb_standard_deviation=2):
        """Scale the current scores by the number of rotational bonds.

        Args:
            nrb_target: Limit for number of rotational bonds.
            nrb_standard_deviation: STD defines the width of the gaussian above the limit nrb_target.
        """
        number_of_rotatable_target_scores = [
            number_of_rotatable_bonds_target_clipped(
                p.rdkit_mol, nrb_target, nrb_standard_deviation
            )
            for p in self.molecules
        ]

        new_scores = [
            score * scale
            for score, scale in zip(
                self.get("score"), number_of_rotatable_target_scores
            )
        ]
        self.setprop("score", new_scores)

    def make_initial_population(self) -> List[Schrock]:
        """Make initial population as a list of Schrocks."""
        population = []
        while len(population) < self.population_size:
            metal = random.choice(metals_list)
            ligands = random.choices(
                ligands_list, k=self.mol_options.individual_type.n_ligands
            )
            cat = self.mol_options.individual_type(metal, ligands)
            # remove duplicates
            if cat not in population:
                population.append(cat)
        return population

    @staticmethod
    def crossover(ind1: Schrock, ind2: Schrock) -> Schrock or None:
        """Crossover the graphs of two ligands of Schrock."""
        ind_type = type(ind1)
        # choose one ligand at random from ind1 and crossover with random ligand from ind2, then replace this ligand in ind1 with new ligand
        ind1_ligands = copy.deepcopy(ind1.ligands)
        new_mol = None
        counter = 0
        while not new_mol:
            idx1 = random.randint(0, len(ind1_ligands) - 1)
            idx2 = random.randint(0, len(ind2.ligands) - 1)
            new_mol = graph_crossover(ind1.ligands[idx1].mol, ind2.ligands[idx2].mol)
            counter += 1
            if counter > 10:
                return None
        try:
            Chem.SanitizeMol(new_mol)
            # this will catch if new_mol has no donor atom
            new_ligand = Ligand(new_mol)
            ind1_ligands[idx1] = new_ligand
            child = ind_type(ind1.metal, ind1_ligands)
            child.assemble()
            return child
        except Exception:
            return None

    @staticmethod
    def mutate(ind: Schrock) -> Schrock or None:
        """Mutate the graph of one ligand of a Schrock."""
        # pick one ligand at random, mutate and replace in ligand list
        idx = random.randint(0, len(ind.ligands) - 1)
        new_mol = None
        counter = 0
        while not new_mol:
            new_mol = graph_mutate(ind.ligands[idx].mol)
            counter += 1
            if counter > 10:
                return None
        try:
            Chem.SanitizeMol(new_mol)
            ind.ligands[idx] = Ligand(new_mol)
            ind.assemble()
            return ind
        except Exception:
            return None

    def calculate_normalized_fitness(self):
        """Normalize the scores to get probabilities for mating selection."""

        # convert to high and low scores.
        scores = [ind.score for ind in self.population]

        max_score = np.nanmax(scores)
        shifted_scores = [
            0 if np.isnan(score) else score - max_score for score in scores
        ]
        sum_scores = sum(shifted_scores)
        if sum_scores == 0:
            print(
                "WARNING: Shifted scores are zero. Normalized fitness is therefore dividing with "
                "zero, could be because the population only contains one individual"
            )

        for individual, shifted_score in zip(self.population, shifted_scores):
            individual.normalized_fitness = abs(shifted_score / sum_scores)

    def save(self, directory=None, name="GA.pkl"):
        """Save instance to file for later retrieval."""
        filename = os.path.join(directory, name)
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def assign_idx(self):
        for i, molecule in enumerate(self.population):
            setattr(molecule, "idx", (self.generation_num, i))

    def debug_run(self):
        self.population = self.make_initial_population()

        self.population = self.calculate_scores(self.population, gen_id=0)

        self.save(directory=".")

    def run(self):

        # self.population = self.data_loader.load_debug()
        # Julius make population
        self.population = self.make_initial_population()

        self.population = self.calculate_scores(self.population, gen_id=0)

        # self.population = self.scorer.scoring_submitter(self.population)

        # Save current population for debugging
        self.save(directory=self.args["output_dir"], name="GA_debug_firstit.pkl")

        # Functionality to check synthetic accessibility
        if self.args["sa_screening"]:
            self.sascorer.get_sa()

        # Normalize the score of population individuals to value between 0 and 1
        self.calculate_normalized_fitness()

        # Save the generation as pickle file
        self.assign_idx()
        self.save(directory=self.args["output_dir"], name="GA00.pkl")

        self.output_handler.write_out(self.population, name="GA0.out")

        logging.info("Finished initial generation")

        # Start evolving
        for generation in range(self.args["generations"]):

            # Counter for tracking generation number
            self.generation_num += 1
            logging.info("Starting generation %d", self.generation_num)

            # If debugging simply reuse previous pop
            if self.args["debug"]:
                children = [
                    Individual(Chem.MolFromSmiles("CCCCN")),
                    Individual(Chem.MolFromSmiles("CCCN")),
                    Individual(Chem.MolFromSmiles("CCN")),
                ]
            else:
                children = self.reproduce(
                    self.args["population_size"],
                    self.args["mutation_rate"],
                    molecule_filter=molecule_filter,
                )

            # Calculate new scores
            logging.info("Getting scores for new population")

            self.children = self.scorer.score_population(children)

            # Save for debugging
            self.output_handler.save(
                children,
                directory=self.args["output_dir"],
                name=f"GA{self.generation_num:02d}_debug2.pkl",
            )
            self.population = self.prune(self.population + self.children)

            # Functionality to compute synthetic accessibility
            if self.args["sa_screening"]:
                self.sascorer.get_sa()

            # Normalize new scores to prep for next gen
            self.calculate_normalized_fitness()

            self.assign_idx()
            self.output_handler.write_out(self.population, name="GA0.out")
            self.save(
                directory=self.args["output_dir"],
                name=f"GA{self.generation_num:02d}.pkl",
            )
            # Save data from current generation
            logging.info(f"Gen No. {self.generation_num} finished")

            # Print gen table to output file
            # TODO
