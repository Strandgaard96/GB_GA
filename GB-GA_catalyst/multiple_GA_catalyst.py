# %%
import copy
import logging
import random
from multiprocessing import Pool
import time
import numpy as np
from rdkit import Chem
import os
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
from catalyst.utils import Generation
import GB_GA as ga
import filters

from GA_catalyst import GA

# %%
# timing_formatter = logging.Formatter('%(message)s')
# timing_logger = logging.getLogger('timing')
# timing_logger.setLevel(logging.INFO)
# timing_file_handler = logging.FileHandler('scoring_timings.log')
# timing_file_handler.setFormatter(timing_formatter)
# timing_logger.addHandler(timing_file_handler)

# warning_formatter = logging.Formatter('%(levelname)s:%(message)s')
# warning_logger = logging.getLogger('warning')
# warning_logger.setLevel(logging.WARNING)
# warning_file_handler = logging.FileHandler('scoring_warning.log')
# warning_file_handler.setFormatter(warning_formatter)
# warning_logger.addHandler(warning_file_handler)
timing_logger = "a"
warning_logger = "b"

# directory = sys.argv[-1]
# n_cpus = int(sys.argv[-2])
# dest_dir = sys.argv[-3]
directory = os.getcwd()
dest_dir = directory
n_cpus = 1

# %%

n_runs = 1
population_size = 50
mating_pool_size = population_size
scoring_script = "/home/julius/soft/GB-GA/catalyst/prereactant_scoring.py"
scaling_function = open_linear_scaling
generations = 30
mutation_rate = 0.5
crossover_rate = 1
co.average_size = 25.0
co.size_stdev = 5.0
n_confs = 5
randomseed = 101
prune_population = True
sa_screening = True


o = 7  # start at #
for m in range(n_runs):
    n = o + m
    file_name = os.path.join(
        "/home/julius/soft/GB-GA/GB-GA_catalyst/chunks", f"pop{n:02d}.smi"
    )
    run_directory = os.path.join(directory, f"run{n:02d}")
    scoring_args = [n_confs, randomseed, timing_logger, warning_logger, run_directory]

    t0 = time.time()

    args = [
        population_size,
        file_name,
        scoring_script,
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
        n,
        dest_dir,
    ]
    # Run the GA
    GA(args)
    t1 = time.time()
    print(f"\n* Total duration: {(t1-t0)/60.0:.2f} minutes \n \n")

# %%
