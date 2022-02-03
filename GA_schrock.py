"""Example Google style docstrings.

This is the driver script for running a GA algorithm on the Schrock catalyst

Example:
    How to run

        $ python GA_schrock.py args

Todo:
    * For module TODOs

"""

import time
import sys
import argparse
import os
import logging
from multiprocessing import Pool
import random
import copy
import numpy as np

# Rdkit stuff
from rdkit import Chem

# Homemade stuff from Julius mostly
import crossover as co
import scoring_functions as sc

from catalyst.ts_scoring import ts_scoring
from catalyst.utils import Generation, mols_from_smi_file
from catalyst.fitness_scaling import (
    scale_scores,
    linear_scaling,
    sigmoid_scaling,
    open_linear_scaling,
    exponential_scaling,
)
from sa import reweigh_scores_by_sa, neutralize_molecules
import GB_GA as ga

# Julius filter functionality.
import filters

molecule_filter = filters.get_molecule_filters(None, "./filters/alert_collection.csv")


def get_arguments(arg_list=None):
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided. Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the arguments

    """
    parser = argparse.ArgumentParser(
        description="Run GA algorithm", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=4,
        help="Sets the size of population pool",
    )
    parser.add_argument(
        "--mating_pool_size",
        type=int,
        default=4,
        help="Size of mating pool",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random initialization of population",
    )
    parser.add_argument(
        "--n_confs",
        type=int,
        default=5,
        help="How many conformers to generate",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2,
        help="How many times is the population optimized",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.5,
        help="Mutation rate",
    )
    parser.add_argument(
        "--prune_population",
        dest="prune_population",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--sa_screening",
        dest="sa_screening",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--n_tries",
        type=int,
        default=2,
        help="How many overall runs of the GA",
    )
    parser.add_argument(
        "--scoring_args",
        type=list,
        default=[],
        help="How many overall runs of the GA",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=6,
        help="Number of cores to distribute over",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="ZINC_first_1000.smi",
        help="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test",
        help="Directory to put various files",
    )
    return parser.parse_args(arg_list)


def GA(args):
    """

    Args:
        args: Dictionary

    Returns:
        gen: Generation class that contains the results of the final generation
    """

    # Make the logger available to this function
    logger = logging.getLogger("my logger")

    # Create initial population and get initial score
    population = ga.make_initial_population(args["population_size"], args["file_name"])
    prescores = sc.calculate_scores(
        population, args["scoring_function"], args["scoring_args"]
    )
    population.setprop("score", prescores)
    population.sortby("score")

    # Functionality to check synthetic accessibility
    if args["sa_screening"]:
        neutralize_molecules(population)
        reweigh_scores_by_sa(population)

    # TODO FILL IN COMMENT
    ga.calculate_normalized_fitness(population)

    # Instantiate generation class for containing the generation results
    gen = Generation(generation_num=0, children=population, survivors=population)
    run_No = 0

    # Save the generation as pickle file.
    gen.save(directory=args["output_dir"], run_No=run_No)
    gen.print()

    # Start the generations based on the initialized population
    for generation in range(args["generations"]):

        # Counter for tracking generation number
        generation_num = generation + 1

        # TODO ADD COMMENT
        population.clean_mutated_survival_and_parents()

        # Making new Children
        mating_pool = ga.make_mating_pool(population, args["mating_pool_size"])
        new_population = ga.reproduce(
            mating_pool,
            args["population_size"],
            args["mutation_rate"],
            filter=molecule_filter,
        )
        new_population.generation_num = generation_num
        new_population.assign_idx()
        population.molecules.sort(key=lambda x: x.rdkit_mol.GetNumAtoms(), reverse=True)

        # Calculate new scores based on new population
        scores = sc.calculate_scores(
            new_population, args["scoring_function"], args["scoring_args"]
        )
        new_population.setprop("score", scores)
        new_population.sortby("score")
        if args["sa_screening"]:
            neutralize_molecules(new_population)
            reweigh_scores_by_sa(new_population)

        # Select best Individuals from old and new population
        potential_survivors = copy.deepcopy(population.molecules)
        for mol in potential_survivors:
            mol.survival_idx = mol.idx
        population = ga.sanitize(
            potential_survivors + new_population.molecules,
            args["population_size"],
            args["prune_population"],
        )

        # SURVIVORS
        population.generation_num = generation_num
        population.assign_idx()
        ga.calculate_normalized_fitness(population)

        # Create generation object from the result. And save for this generation
        gen = Generation(
            generation_num=generation_num, children=new_population, survivors=population
        )
        gen.save(directory=args["output_dir"], run_No=run_No)
        # Awesome print functionality by julius that format some results as nice table in log file.
        gen.print()

    return gen


def main():
    args = get_arguments()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Variables for crossover module
    co.average_size = 25.0
    co.size_stdev = 5.0
    n_tries = args.n_tries
    seeds = np.random.randint(100_000, size=2 * n_tries)

    # Log the argparse set values
    logging.info("Input args: %r", args)

    # Parse the set arguments and add scoring function to dict.
    args_dict = vars(args)
    args_dict["scoring_function"] = sc.logP_max
    # Create list of dicts for the distributed GAs
    GA_args = [args_dict for i in range(n_tries)]

    # For debugging GA to prevent multiprocessing cluttering the traceback
    # generations = GA(GA_args[0])

    # Start the time
    t0 = time.time()
    # Run the GA
    with Pool(args.n_cpus) as pool:
        generations = pool.map(GA, GA_args)

    for gen in generations:
        print(gen)
    t1 = time.time()
    logging.info(f"# Total duration: {(t1 - t0) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
