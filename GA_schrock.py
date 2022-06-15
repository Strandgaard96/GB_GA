"""

This is the driver script for running a GA algorithm on the Schrock catalysts

Example:
    How to run:

        $ python GA_schrock.py --args

"""

import pathlib
import time
import argparse
import os
import copy
from pathlib import Path
import sys
from copy import deepcopy
import logging

# Homemade stuff from Julius mostly
import crossover as co
from scoring import scoring_functions as sc

from scoring.scoring import (
    rdkit_embed_scoring,
    rdkit_embed_scoring_NH3toN2,
    rdkit_embed_scoring_NH3plustoNH3,
)
from my_utils.my_utils import (
    Generation,
    Population,
    Individual,
    get_git_revision_short_hash,
)
from sa.neutralize import neutralize_molecules
from sa.sascorer import reweigh_scores_by_sa
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
        default=100,
        help="Sets the size of population pool.",
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
        "--n_tries",
        type=int,
        default=1,
        help="How many overall runs of the GA",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=2,
        help="Number of cores to distribute over",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=1,
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
        "--file_name",
        type=str,
        default="data/ZINC_250k.smi",
        help="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generation_debug",
        help="Directory to put various files",
    )
    parser.add_argument(
        "--database",
        type=pathlib.Path,
        default="ase_database.db",
        help="Path to database to write files to",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--write_db", action="store_true")
    parser.add_argument("--supress_amines", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="2",
        help="gfn method to use",
    )
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument(
        "--scoring_func",
        dest="func",
        choices=[
            "rdkit_embed_scoring",
            "rdkit_embed_scoring_NH3toN2",
            "rdkit_embed_scoring_NH3plustoNH3",
        ],
        required=True,
        help="""Choose one of the specified scoring functions to be run.""",
    )
    return parser.parse_args(arg_list)


def GA(args):
    """

    Args:
        args(dict): Dictionary containint all relevant args for the functionscoring_a

    Returns:
        gen: Generation class that contains the results of the final generation
    """

    # Create initial population and get initial score
    if args["debug"]:
        population = ga.make_initial_population_debug(
            4, "data/ZINC_1000_amines.smi", rand=True
        )
    else:
        population = ga.make_initial_population(
            args["population_size"], args["file_name"], rand=True
        )

    results = sc.slurm_scoring(args["scoring_function"], population, args)
    energies = [res[0] for res in results]
    geometries = [res[1] for res in results]
    geometries2 = [res[2] for res in results]
    min_conf = [res[3] for res in results]

    population.setprop("energy", energies)
    population.setprop("pre_score", energies)
    population.setprop("structure", geometries)
    population.setprop("structure2", geometries2)
    population.setprop("min_conf", min_conf)

    population.setprop("score", energies)
    # Functionality to check synthetic accessibility
    if args["sa_screening"]:
        neutralize_molecules(population)
        reweigh_scores_by_sa(population)

    population.sortby("score")

    # Normalize the score of population infividuals to value between 0 and 1
    ga.calculate_normalized_fitness(population)

    # Instantiate generation class for containing the generation results
    gen = Generation(generation_num=0, children=population, survivors=population)

    # Save the generation as pickle file.
    gen.save(directory=args["output_dir"], run_No=0)
    gen.print()
    with open(args["output_dir"] + "/GA0.out", "w") as f:
        f.write(gen.print(pass_text=True))
        f.write("\n")
        f.write(gen.print(population="children", pass_text=True))
        f.write("\n")
        f.write(gen.summary())

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
        new_population.modify_population(supress_amines=args["supress_amines"])

        # Assign generation and population idx to the population
        new_population.generation_num = generation_num
        new_population.assign_idx()

        # Sort population based on size
        population.molecules.sort(key=lambda x: x.rdkit_mol.GetNumAtoms(), reverse=True)

        # Calculate new scores based on new population
        logging.info("Getting scores for new population")
        results = sc.slurm_scoring(args["scoring_function"], new_population, args)

        energies = [res[0] for res in results]
        geometries = [res[1] for res in results]
        geometries2 = [res[2] for res in results]
        min_conf = [res[3] for res in results]

        new_population.setprop("energy", energies)
        new_population.setprop("pre_score", energies)
        new_population.setprop("structure", geometries)
        new_population.setprop("structure2", geometries2)
        new_population.setprop("min_conf", min_conf)

        new_population.setprop("score", energies)
        # Functionality to check synthetic accessibility
        if args["sa_screening"]:
            neutralize_molecules(new_population)
            reweigh_scores_by_sa(new_population)

        # Sort scores, possibly scaled by SA screening
        new_population.sortby("score")

        # Select best Individuals from old and new population
        potential_survivors = copy.deepcopy(population.molecules)
        for mol in potential_survivors:
            mol.survival_idx = mol.idx

        # The calculated population is merged with current top population
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
        gen = Generation(
            generation_num=generation_num,
            children=new_population,
            survivors=population,
        )
        # Save data from current generation
        logging.info("Saving current generation")
        gen.save(directory=args["output_dir"], run_No=generation_num)

        # Print gen table to output file
        gen.print()
        gen.summary()

        # Print to individual generation files to keep track on the fly
        with open(args["output_dir"] + f"/GA{generation_num}.out", "w") as f:
            f.write(gen.print(pass_text=True))
            f.write("\n")
            f.write(gen.print(population="children", pass_text=True))
            f.write("\n")
            f.write(gen.summary())

    return gen


def main():

    args = get_arguments()

    # Create output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Variables for crossover module

    co.average_size = 50
    co.size_stdev = 10

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

    # Log current git commit
    logging.info("Current git hash: %s", get_git_revision_short_hash())

    # Log the argparse set values
    logging.info("Input args: %r", args)

    # a dictionary mapping strings of function names to function objects:
    funcs = {
        "rdkit_embed_scoring": rdkit_embed_scoring,
        "rdkit_embed_scoring_NH3toN2": rdkit_embed_scoring_NH3toN2,
        "rdkit_embed_scoring_NH3plustoNH3": rdkit_embed_scoring_NH3plustoNH3,
    }

    # Get arguments as dict and add scoring function to dict.
    args_dict = vars(args)
    args_dict["scoring_function"] = funcs[args.func]

    # Create list of dicts for the distributed GAs
    GA_args = args_dict

    # Start the time
    t0 = time.time()

    # Run the GA
    generations = GA(GA_args)

    # Final output handling and logging
    t1 = time.time()
    logging.info(f"# Total duration: {(t1 - t0) / 60.0:.2f} minutes")

    # Addded this to return to the commandline if running this driver
    # on the frontend.
    sys.exit(0)


if __name__ == "__main__":
    main()
    # Load the children population to compare SA scores
    # import pickle
    # with open("/home/magstr/generation_data/supress3/GA07.pkl", "rb") as f:
    #     gen = pickle.load(f)
    #
    # from rdkit import Chem
    # initial_population = gen.survivors
    # l = initial_population.get("pre_score")
    # l.append(1)
    # #ligand = Chem.MolFromSmiles('CC1=C(OC2CCCC2)C=C([C@H](C)NN)C1')
    # ligand = Chem.MolFromSmiles(Chem.MolToSmiles(gen.children.molecules[26].rdkit_mol))
    # ligand = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.AddHs(ligand)))
    # cut_idx = [[0]]
    # pop = Population()
    # pop.molecules.append(Individual(ligand, cut_idx=cut_idx[0][0], score=1))
    # pop.setprop('pre_score',[1])
    # tmp = Chem.MolFromSmiles(Chem.MolToSmiles(gen.children.molecules[26].rdkit_mol))
    # pop.molecules.append(Individual(tmp, cut_idx=gen.children.molecules[9].cut_idx, score=gen.children.molecules[9].score))
    # #initial_population.generation_num = 0
    # #initial_population.assign_idx()
    # pop.setprop("pre_score", [1,1])
    # #initial_population.setprop("score", [1])
    # neutralize_molecules(pop)
    # reweigh_scores_by_sa(pop)
    # print(pop.molecules[0].sa_score)
    # print(pop.molecules[1].sa_score)
