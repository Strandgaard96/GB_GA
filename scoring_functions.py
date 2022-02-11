"""
Written by Jan H. Jensen 2018
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import rdFMCS

from rdkit import rdBase

rdBase.DisableLog("rdApp.error")

import numpy as np
import sys
from multiprocessing import Pool
import subprocess
import os
import shutil
import string
import random
import pickle
import time
import submitit
import sascorer

from catalyst.utils import Population

logP_values = np.loadtxt("data/logP_values.txt")
SA_scores = np.loadtxt("data/SA_scores.txt")
cycle_scores = np.loadtxt("data/cycle_scores.txt")
SA_mean = np.mean(SA_scores)
SA_std = np.std(SA_scores)
logP_mean = np.mean(logP_values)
logP_std = np.std(logP_values)
cycle_mean = np.mean(cycle_scores)
cycle_std = np.std(cycle_scores)


def wait_for_jobs_to_finish(job_ids):
    """
    This script checks with slurm if a specific set of jobids is finished with a
    frequency of 1 minute.
    Stops when the jobs are done.
    """
    while True:
        job_info1 = os.popen("squeue -p mko").readlines()[1:]
        job_info2 = os.popen("squeue -a -u julius").readlines()[1:]
        current_jobs1 = {int(job.split()[0]) for job in job_info1}
        current_jobs2 = {int(job.split()[0]) for job in job_info2}
        current_jobs = current_jobs1 | current_jobs2
        if current_jobs.isdisjoint(job_ids):
            break
        else:
            time.sleep(20)


def slurm_scoring(sc_function, population, scoring_args):
    """Evaluates a scoring function for population on SLURM cluster
    Args:
        sc_function (function): Scoring function which takes molecules and id (int,int) as input
        population (List): List of rdkit Molecules
        ids (List of Tuples of Int): Index of each molecule (Generation, Individual)
    Returns:
        List: List of results from scoring function
    """
    executor = submitit.AutoExecutor(
        folder="scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"sc_g{population.molecules[0].idx[0]}",
        cpus_per_task=scoring_args["cpus_per_task"],
        slurm_mem_per_cpu="1GB",
        timeout_min=30,
        slurm_partition="kemi1",
        slurm_array_parallelism=100,
    )

    # Extract ids
    ids = []
    for ind in population.molecules:
        ids.append(ind.idx)

    # cpus per task needs to be list
    cpus_per_task_list = [scoring_args["cpus_per_task"] for p in population.molecules]
    n_confs_list = [scoring_args["n_confs"] for p in population.molecules]
    cleanup_list = [scoring_args["cleanup"] for p in population.molecules]
    jobs = executor.map_array(
        sc_function,
        population.molecules,
        ids,
        cpus_per_task_list,
        n_confs_list,
        cleanup_list,
    )

    results = [
        catch(job.result, handle=lambda e: (np.nan, None)) for job in jobs
    ]  # catch submitit exceptions and return same output as scoring function (np.nan, None) for (energy, geometry)
    if scoring_args["cleanup"]:
        shutil.rmtree("scoring_tmp")
    return results


def catch(func, *args, handle=lambda e: e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)

    target = args[0]
    try:
        mcs = rdFMCS.FindMCS(
            [mol, target],
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        score = mcs.numAtoms / target.GetNumAtoms()
        return score

    except:
        print("Failed ", Chem.MolToSmiles(mol))
        return None
