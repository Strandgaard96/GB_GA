"""

This is the driver script for running a GA algorithm on the Schrock catalysts

Example:
    How to run:

        $ python GA_schrock.py args

"""

import time
import argparse
import os
import copy
from pathlib import Path
import sys

# Homemade stuff from Julius mostly
import crossover as co
from scoring import scoring_functions as sc

from scoring.scoring import rdkit_embed_scoring
from my_utils.my_utils import Generation
import logging
from sa import reweigh_scores_by_sa, neutralize_molecules
import GB_GA as ga

# Julius filter functionality.
import filters

molecule_filter = filters.get_molecule_filters(None, "./filters/alert_collection.csv")


def get_arguments(arg_list=None):
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided.
        Otherwise default arguments are used

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
        default=3,
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
        default=1,
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
        default=1,
        help="How many overall runs of the GA",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=1,
        help="Number of cores to distribute over",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="data/ZINC_1000_amines.smi",
        help="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generation_debug",
        help="Directory to put various files",
    )
    return parser.parse_args(arg_list)


def get_scoring_args(args):

    scoring_args = {}

    scoring_args["n_confs"] = args.n_confs
    scoring_args["cpus_per_task"] = args.n_cpus
    scoring_args["cleanup"] = False
    scoring_args["output_dir"] = args.output_dir
    return scoring_args


def GA(args):
    """

    Args:
        args(dict): Dictionary containint all relevant args for the functionscoring_a

    Returns:
        gen: Generation class that contains the respickleults of the final generation
    """

    # Create initial population and get initial score
    population = ga.make_initial_population_res(
        args["population_size"], args["file_name"], rand=True
    )

    results = sc.slurm_scoring(
        args["scoring_function"], population, args["scoring_args"]
    )
    energies = [res[0] for res in results]
    geometries = [res[1] for res in results]
    min_conf = [res[2] for res in results]

    population.setprop("energy", energies)
    population.setprop("score", energies)
    population.setprop("min_conf", min_conf)

    population.sortby("score")

    # Functionality to check synthetic accessibility
    if args["sa_screening"]:
        neutralize_molecules(population)
        reweigh_scores_by_sa(population)

    # Normalize the score of population infividuals to value between 0 and 1
    ga.calculate_normalized_fitness(population)

    # Instantiate generation class for containing the generation results
    gen = Generation(generation_num=0, children=population, survivors=population)

    # Save the generation as pickle file.
    gen.save(directory=args["output_dir"], run_No=0)
    gen.print()

    logging.info("Finished initial generation")

    # Start the generations based on the initialized population
    for generation in range(args["generations"]):

        # Counter for tracking generation number
        generation_num = generation + 1
        logging.info("Starting generation %d", generation_num)

        # I think this takes the population and reset some Individuals' attributes
        # Such that they can be set in new generation.
        population.clean_mutated_survival_and_parents()

        # Making new Children
        mating_pool = ga.make_mating_pool(population, args["mating_pool_size"])
        new_population = ga.reproduce(
            mating_pool,
            args["population_size"],
            args["mutation_rate"],
            molecule_filter=molecule_filter,
        )

        # Ensures that new molecules have a primary amine attachment point.
        logging.info("Creating attachment points for new population")
        new_population.modify_population()

        # Assign which generation the population is form.
        new_population.generation_num = generation_num
        # Assign idx to molecules in population that contain the index in population,
        # but also the generation each molecule comes from
        new_population.assign_idx()

        # Sort population based on size
        population.molecules.sort(key=lambda x: x.rdkit_mol.GetNumAtoms(), reverse=True)

        # Calculate new scores based on new population
        logging.info("Getting scores for new population")
        results = sc.slurm_scoring(
            args["scoring_function"], new_population, args["scoring_args"]
        )

        energies = [res[0] for res in results]
        geometries = [res[1] for res in results]

        new_population.setprop("energy", energies)
        new_population.setprop("score", energies)
        new_population.sortby("score")

        if args["sa_screening"]:
            neutralize_molecules(new_population)
            reweigh_scores_by_sa(new_population)

        # Select best Individuals from old and new population
        potential_survivors = copy.deepcopy(population.molecules)
        for mol in potential_survivors:
            mol.survival_idx = mol.idx

        # Here the total population of new and old are sorted according to score, and the
        # Remaining population is the ones with thte highest scores
        # Note that there is a namechange here. new_population is merged with population
        # which is effectively the new population.
        population = ga.sanitize(
            potential_survivors + new_population.molecules,
            args["population_size"],
            args["prune_population"],
        )

        # Population now contains survivors after sanitation.
        population.generation_num = generation_num

        # Normalize new scores
        ga.calculate_normalized_fitness(population)

        # Create generation object from the result. And save for this generation
        # Here new_population is the generated children. Not all of these are passed to the
        # next generation which is held by survivors.
        gen = Generation(
            generation_num=generation_num, children=new_population, survivors=population
        )
        # Save data from current generation
        logging.info("Saving current generation")
        gen.save(directory=args["output_dir"], run_No=generation_num)

        gen.print()

    return gen


def main():

    args = get_arguments()

    # Create output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Variables for crossover module
    co.average_size = 20
    co.size_stdev = 5

    # How many times to run the GA.
    n_tries = args.n_tries

    # Setup logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),  # For debugging. Can be removed on remote
        ],
    )
    # Log the argparse set values
    logging.info("Input args: %r", args)

    # Parse the set arguments and add scoring function to dict.
    args_dict = vars(args)
    args_dict["scoring_function"] = rdkit_embed_scoring
    args_dict["scoring_args"] = get_scoring_args(args)

    # Create list of dicts for the distributed GAs
    GA_args = args_dict

    # Start the time
    t0 = time.time()

    # Run the GA
    generations = GA(GA_args)

    # Final output handling and logging
    generations.print()
    t1 = time.time()
    logging.info(f"# Total duration: {(t1 - t0) / 60.0:.2f} minutes")

    # Addded this to return to the commandline if running this driver
    # on the frontend.
    sys.exit(0)


if __name__ == "__main__":
    main()
