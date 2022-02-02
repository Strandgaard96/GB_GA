# %%
import copy
import logging
import random
from multiprocessing import Pool
import time
import numpy as np
from rdkit import Chem
import sys

sys.path.append("/home/julius/soft/GB-GA/")
import crossover as co
import scoring_functions as sc
from sa import reweigh_scores_by_sa, neutralize_molecules
from catalyst.fitness_scaling import (
    scale_scores,
    linear_scaling,
    sigmoid_scaling,
    open_linear_scaling,
    exponential_scaling,
)
from catalyst.ts_scoring import ts_scoring
from catalyst.path_scoring import path_scoring
from catalyst.prereactant_scoring import prereactant_scoring
from catalyst.utils import Generation, mols_from_smi_file
import GB_GA as ga
import filters


molecule_filter = filters.get_molecule_filters(
    None, "/home/julius/soft/GB-GA/filters/alert_collection.csv"
)


# %%
def GA(args):
    (
        population_size,
        file_name,
        scoring_function,
        scaling_function,
        generations,
        mating_pool_size,
        mutation_rate,
        crossover_rate,
        scoring_args,
        prune_population,
        n_cpus,
        sa_screening,
        randomseed,
        run,
        dest_dir,
    ) = args

    cpus_per_worker = int(np.ceil(n_cpus / population_size))
    np.random.seed(randomseed)
    random.seed(randomseed)
    population = ga.make_initial_population(population_size, file_name, rand=False)
    # sort molecules descending in size
    population.molecules.sort(key=lambda x: x.rdkit_mol.GetNumAtoms(), reverse=True)
    sc.calculate_scores_parallel(
        population=population,
        function=scoring_function,
        scoring_args=scoring_args,
        n_cpus=n_cpus,
        cpus_per_worker=cpus_per_worker,
    )

    if sa_screening:
        neutralize_molecules(population)
        reweigh_scores_by_sa(population)

    scale_scores(population, scaling_function, sa_screening=sa_screening)
    population.sortby("score")

    ga.calculate_normalized_fitness(population)
    gen = Generation(generation_num=0, children=population, survivors=population)
    gen.save(dest_dir, run=run)
    gen.print()
    stagnation_count = 0
    stagnation = False

    for generation in range(generations):
        generation_num = generation + 1
        population.clean_mutated_survival_and_parents()
        # Making new Children
        mating_pool = ga.make_mating_pool(population, mating_pool_size)
        # new_population = ga.reproduce2(
        #     mating_pool, population_size, mutation_rate, crossover_rate ,filter=molecule_filter)
        new_population = ga.reproduce(
            mating_pool, population_size, mutation_rate, filter=molecule_filter
        )
        new_population.generation_num = generation_num
        new_population.assign_idx()
        population.molecules.sort(key=lambda x: x.rdkit_mol.GetNumAtoms(), reverse=True)
        sc.calculate_scores_parallel(
            population=new_population,
            function=scoring_function,
            scoring_args=scoring_args,
            n_cpus=n_cpus,
            cpus_per_worker=cpus_per_worker,
        )
        if sa_screening:
            neutralize_molecules(new_population)
            reweigh_scores_by_sa(new_population)

        scale_scores(new_population, scaling_function, sa_screening=sa_screening)
        new_population.sortby("score")

        # Select best Individuals from old and new population
        potential_survivors = copy.deepcopy(population.molecules)
        for mol in potential_survivors:
            mol.survival_idx = mol.idx
        population = ga.sanitize(
            potential_survivors + new_population.molecules,
            population_size,
            prune_population,
        )  # SURVIVORS
        population.generation_num = generation_num
        population.assign_idx()
        ga.calculate_normalized_fitness(population)

        gen = Generation(
            generation_num=generation_num, children=new_population, survivors=population
        )

        # num_new_children = len([x.parentB_idx for x in gen.survivors.molecules if type(x.parentB_idx) == tuple])

        # if not stagnation:
        #     if num_new_children < population_size*0.1:
        #         stagnation_count += 1
        #     else:
        #         stagnation_count = 0
        #     if stagnation_count == 3:
        #         stagnation = True

        # else:
        #     crossover_rate = 0
        #     mutation_rate = 1

        gen.save(dest_dir, run=run)
        gen.print()


# %%
if __name__ == "__main__":

    randomseed = 9
    random.seed(randomseed)
    np.random.seed(randomseed)

    timing_formatter = logging.Formatter("%(message)s")
    timing_logger = logging.getLogger("timing")
    timing_logger.setLevel(logging.INFO)
    timing_file_handler = logging.FileHandler("scoring_timings.log")
    timing_file_handler.setFormatter(timing_formatter)
    timing_logger.addHandler(timing_file_handler)

    warning_formatter = logging.Formatter("%(levelname)s:%(message)s")
    warning_logger = logging.getLogger("warning")
    warning_logger.setLevel(logging.WARNING)
    warning_file_handler = logging.FileHandler("scoring_warning.log")
    warning_file_handler.setFormatter(warning_formatter)
    warning_logger.addHandler(warning_file_handler)

    directory = sys.argv[-1]
    n_cpus = int(sys.argv[-2])
    dest_dir = sys.argv[-3]
    # directory = '.'
    # n_cpus = 1

    scoring_function = prereactant_scoring
    scaling_function = open_linear_scaling

    n_confs = 1
    scoring_args = [n_confs, randomseed, timing_logger, warning_logger, directory]

    molecule_filter = filters.get_molecule_filters(
        None, "/home/julius/soft/GB-GA/filters/alert_collection.csv"
    )
    file_name = "/home/julius/soft/GB-GA/ZINC_1000_amines.smi"
    population_size = 8
    mating_pool_size = population_size
    generations = 8
    mutation_rate = 0.5
    crossover_rate = 1
    co.average_size = 25.0
    co.size_stdev = 5.0
    prune_population = True
    n_tries = 1
    sa_screening = True
    seeds = np.random.randint(100000, size=2 * n_tries)
    cpus_per_worker = int(np.ceil(n_cpus / population_size))
    print("* scoring_function", scoring_function)
    print("* scaling_function", scaling_function)
    print("* SA screening", sa_screening)
    print("* filters", molecule_filter)
    print("* n_confs", n_confs)
    print("* population_size", population_size)
    print("* mating_pool_size", mating_pool_size)
    print("* generations", generations)
    print("* mutation_rate", mutation_rate)
    print("* average_size/size_stdev", co.average_size, co.size_stdev)
    print("* prune population", prune_population)
    print("* number of tries", n_tries)
    print("* number of CPUs", n_cpus)
    print("* number of CPUs per worker", cpus_per_worker)
    print("* SMILES input file", file_name)
    print("* seeds seed", randomseed)
    print("* seeds", ",".join(map(str, seeds)))
    print("")

    # Get Timings
    timing_logger.info(f"# Running on {n_cpus} cores")

    t0 = time.time()

    args = [
        population_size,
        file_name,
        scoring_function,
        scaling_function,
        generations,
        mating_pool_size,
        mutation_rate,
        crossover_rate,
        scoring_args,
        prune_population,
        n_cpus,
        sa_screening,
        randomseed,
        0,
        dest_dir,
    ]
    # Run the GA
    GA(args)

    t1 = time.time()
    print(f"\n* Total duration: {(t1-t0)/60.0:.2f} minutes")


# %%
