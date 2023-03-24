"""Written by Jan H. and modified by Magnus Strandgaard 2023.

Jensen 2018.
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
"""

import logging
import math
import os
import pickle
import random
from abc import ABC

import numpy as np
from rdkit import Chem

import crossover as co
import mutate as mu
from utils.classes import DataLoader, Individual, OutputHandler, Scoring
from utils.sa import SaScorer


class GeneticAlgorithm(ABC):
    """

    Args:
        args(dict): Dictionary containing all the commandline input args.

    Returns:
        gen: MoleculeHandler class that contains the results of the final generation
    """

    def __init__(self, args):
        self.args = args
        self.data_loader = DataLoader(args)
        self.scorer = Scoring(args)
        self.sascorer = SaScorer()
        self.output_handler = OutputHandler()

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
            reverse=True,
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
            individual.normalized_fitness = shifted_score / sum_scores

    def save(self, directory=None, name="GA.pkl"):
        """Save instance to file for later retrieval."""
        filename = os.path.join(directory, name)
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def run(self):

        generation_num = 0

        self.population = self.data_loader.load_data()

        self.population = self.scorer.score_population(self.population)

        # Save current population for debugging
        self.save(directory=self.args["output_dir"], name="GA_debug_firstit.pkl")

        # Functionality to check synthetic accessibility
        if self.args["sa_screening"]:
            self.sascorer.get_sa()

        # Normalize the score of population individuals to value between 0 and 1
        self.calculate_normalized_fitness()

        # Save the generation as pickle file and print current output
        self.save(directory=self.args["output_dir"], name="GA00.pkl")

        logging.info("Finished initial generation")

        # Start evolving
        for generation in range(self.args["generations"]):

            # Counter for tracking generation number
            generation_num += 1
            logging.info("Starting generation %d", generation_num)

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
                name=f"GA{generation_num:02d}_debug2.pkl",
            )
            self.population = self.prune(self.population + self.children)

            # Functionality to compute synthetic accessibility
            if self.args["sa_screening"]:
                self.sascorer.get_sa()

            # Normalize new scores to prep for next gen
            self.calculate_normalized_fitness()

            self.save(
                directory=self.args["output_dir"], name=f"GA{generation_num:02d}.pkl"
            )
            # Save data from current generation
            logging.info(f"Gen No. {generation_num} finished")

            # Print gen table to output file
            # TODO
