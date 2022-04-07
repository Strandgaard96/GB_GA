"""
Written by Jan H. Jensen 2018
"""

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdFMCS

rdBase.DisableLog("rdApp.error")

import numpy as np
import os
import shutil
import time
import submitit
from pathlib import Path

logP_values = np.loadtxt("data/logP_values.txt")
SA_scores = np.loadtxt("data/SA_scores.txt")
cycle_scores = np.loadtxt("data/cycle_scores.txt")
SA_mean = np.mean(SA_scores)
SA_std = np.std(SA_scores)
logP_mean = np.mean(logP_values)
logP_std = np.std(logP_values)
cycle_mean = np.mean(cycle_scores)
cycle_std = np.std(cycle_scores)


def slurm_scoring(sc_function, population, scoring_args):
    """Evaluates a scoring function for population on SLURM cluster
    Args:
        sc_function (function): Scoring function which takes molecules and id (int,int) as input
        population (List): List of rdkit Molecules
        scoring_args (dict):
    Returns:
        results (List): List of results from scoring function
    """
    executor = submitit.AutoExecutor(
        folder=Path(scoring_args["output_dir"]) / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"sc_g{population.molecules[0].idx[0]}",
        cpus_per_task=scoring_args["cpus_per_task"],
        slurm_mem_per_cpu="2GB",
        timeout_min=8,
        slurm_partition="kemi1",
        slurm_array_parallelism=50,
    )

    # Extract ids
    ids = []
    for ind in population.molecules:
        ids.append(ind.idx)

    # cpus per task needs to be list
    cpus_per_task_list = [scoring_args["cpus_per_task"] for p in population.molecules]
    n_confs_list = [scoring_args["n_confs"] for p in population.molecules]
    cleanup_list = [scoring_args["cleanup"] for p in population.molecules]
    output_dir_list = [scoring_args["output_dir"] for p in population.molecules]
    jobs = executor.map_array(
        sc_function,
        population.molecules,
        ids,
        cpus_per_task_list,
        n_confs_list,
        cleanup_list,
        output_dir_list,
    )

    results = [catch(job.result, handle=lambda e: (np.nan, None)) for job in jobs]
    # catch submitit exceptions and return same output as scoring function
    # (np.nan, None) for (energy, geometry)
    if scoring_args["cleanup"]:
        shutil.rmtree("scoring_tmp")
    return results


def catch(func, *args, handle=lambda e: e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)
