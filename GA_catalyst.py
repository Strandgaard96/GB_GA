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
import GB_GA as ga

# Rdkit stuff
from rdkit import Chem

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
        default=15,
        help="How many conformers to generate",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
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
        default=False,
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
        default=4,
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
    scores = prescores

    if sa_screening:
        scores, sascores = reweigh_scores_by_sa(
            neutralize_molecules(population), scores
        )

    fitness = ga.calculate_normalized_fitness(scores)

    if sa_screening:
        print(
            f"{list(zip(scores, prescores, sascores, [Chem.MolToSmiles(mol) for mol in population]))}"
        )
    else:
        print(f"{list(zip(scores, [Chem.MolToSmiles(mol) for mol in population]))}")

    high_scores = []
    for generation in range(generations):
        mating_pool = ga.make_mating_pool(population, fitness, mating_pool_size)
        new_population = ga.reproduce(mating_pool, population_size, mutation_rate)
        new_prescores = sc.calculate_scores_parallel(
            new_population, scoring_function, scoring_args, n_cpus
        )

        if sa_screening:
            new_scores, new_sascores = reweigh_scores_by_sa(
                neutralize_molecules(new_population), new_prescores
            )
            population, scores, prescores, sascores = ga.sanitize(
                population + new_population,
                scores + new_scores,
                population_size,
                prune_population,
                sa_screening,
                prescores + new_prescores,
                sascores + new_sascores,
            )
        else:
            new_scores = new_prescores
            population, scores = ga.sanitize(
                population + new_population,
                scores + new_scores,
                population_size,
                prune_population,
                sa_screening,
            )

        fitness = ga.calculate_normalized_fitness(scores)

        high_scores.append((scores[0], Chem.MolToSmiles(population[0])))

        if sa_screening:
            print(
                f"{list(zip(scores, prescores, sascores, [Chem.MolToSmiles(mol) for mol in population]))}"
            )
        else:
            print(f"{list(zip(scores, [Chem.MolToSmiles(mol) for mol in population]))}")

    return (scores, population, high_scores)


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

    out = GA(args_pass[0])

    with Pool(args.n_cpus) as pool:
        output = pool.map(GA, args_pass)

    for i in range(n_tries):
        (scores, population, high_scores) = output[i]
        all_scores.append(scores)
        results.append(scores[0])
        generations_list.append(high_scores)

    t1 = time.time()
    print(f"# Total duration: {(t1 - t0) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
