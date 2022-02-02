import numpy as np
import time
import crossover as co
import scoring_functions as sc
import sys
import argparse
import os
import logging
from multiprocessing import Pool
import random
import copy
import GB_GA as ga

# Rdkit stuff
from rdkit import Chem

import filters
molecule_filter = filters.get_molecule_filters(None, './filters/alert_collection.csv')

from catalyst.utils import Generation, mols_from_smi_file
from catalyst.fitness_scaling import scale_scores, linear_scaling, sigmoid_scaling, open_linear_scaling, exponential_scaling
from sa import reweigh_scores_by_sa, neutralize_molecules


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
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
        default=3,
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
        default=".",
        help="Directory to put various files",
    )
    return parser.parse_args(arg_list)


def GA(args):
    (
        population_size,
        file_name,
        scoring_function,
        scaling_function,
        generations,
        mating_pool_size,
        mutation_rate,
        scoring_args,
        prune_population,
        n_cpus,
        sa_screening,
        seed,
    ) = args

    np.random.seed(seed)
    random.seed(seed)

    population = ga.make_initial_population(population_size, file_name)
    prescores = sc.calculate_scores(population, scoring_function, scoring_args)
    # scores = normalize(prescores)
    population.setprop('score',prescores)

    if sa_screening:
        neutralize_molecules(population)
        reweigh_scores_by_sa(population)

    # TODO THIS DOES NTO WORK. The scaling sets scores to zero
    #scale_scores(population, scaling_function, sa_screening=sa_screening)
    population.sortby('score')

    ga.calculate_normalized_fitness(population)

    gen = Generation(generation_num=0, children=population,
                     survivors=population)
    dest_dir='test'
    run=0
    gen.save(dest_dir, run=run)
    gen.print()

    for generation in range(generations):
        generation_num = generation + 1
        population.clean_mutated_survival_and_parents()
        # Making new Children
        mating_pool = ga.make_mating_pool(population, mating_pool_size)
        # new_population = ga.reproduce2(
        #     mating_pool, population_size, mutation_rate, crossover_rate ,filter=molecule_filter)
        new_population = ga.reproduce(
            mating_pool, population_size, mutation_rate, filter=molecule_filter)
        new_population.generation_num = generation_num
        new_population.assign_idx()
        population.molecules.sort(
            key=lambda x: x.rdkit_mol.GetNumAtoms(), reverse=True)


        scores = sc.calculate_scores(population, scoring_function, scoring_args)
        population.setprop('score', scores)
        if sa_screening:
            neutralize_molecules(new_population)
            reweigh_scores_by_sa(new_population)

        #scale_scores(new_population, scaling_function,
        #            sa_screening=sa_screening)
        new_population.sortby('score')

        # Select best Individuals from old and new population
        potential_survivors = copy.deepcopy(population.molecules)
        for mol in potential_survivors:
            mol.survival_idx = mol.idx
        population = ga.sanitize(potential_survivors + new_population.molecules,
                                 population_size, prune_population)  # SURVIVORS
        population.generation_num = generation_num
        population.assign_idx()
        ga.calculate_normalized_fitness(population)

        gen = Generation(generation_num=generation_num,
                         children=new_population, survivors=population)

        gen.save(dest_dir, run=run)
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
    # logging.debug('This is a debug message')
    # logging.info('This is an info message')
    # logging.warning('This is a warning message')
    # logging.error('This is an error message')
    # logging.critical('This is a critical message')

    # Default values from original code
    scoring_function = sc.logP_max
    scaling_function = open_linear_scaling

    scoring_args = []
    co.average_size = 25.0
    co.size_stdev = 5.0
    n_tries = args.n_tries
    seeds = np.random.randint(100_000, size=2 * n_tries)

    #
    logging.info(
        """\nInitializing GA with args:
                        n_confs %d,
                        population_size %d
                        mating_pool_size %d
                        generations %d
                        mutation_rate %d
                        average_size/size_stdev %d %d
                        initial pool %s
                        prune population %s
                        number of tries %d
                        number of CPUs %d
                        inpput file %s
                        SA screening %s
                        seed %d
    """,
        args.n_confs,
        args.population_size,
        args.mating_pool_size,
        args.generations,
        args.mutation_rate,
        co.average_size,
        co.size_stdev,
        args.file_name,
        args.prune_population,
        args.n_tries,
        args.n_cpus,
        args.file_name,
        args.sa_screening,
        args.seed,
    )

    results = []
    t0 = time.time()
    all_scores = []
    generations_list = []

    index = slice(0, n_tries) if args.prune_population else slice(n_tries, 2 * n_tries)

    temp_args = [
        [
            args.population_size,
            args.file_name,
            scoring_function,
            scaling_function,
            args.generations,
            args.mating_pool_size,
            args.mutation_rate,
            scoring_args,
            args.prune_population,
            args.n_cpus,
            args.sa_screening,
        ]
        for i in range(n_tries)
    ]
    args_pass = []
    for x, y in zip(temp_args, seeds[index]):
        x.append(y)
        args_pass.append(x)

    # Run the GA
    with Pool(args.n_cpus) as pool:
        generations = pool.map(GA, args_pass)

    for gen in generations:
        print(gen)
    t1 = time.time()
    print(f"# Total duration: {(t1 - t0) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
