# -*- coding: utf-8 -*-
"""
Module that contains submitit scoring functions. These submit multiple
scoring calculation jobs.
"""

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdFMCS

# rdBase.DisableLog("rdApp.error")

import numpy as np
import os
import shutil
import time
import submitit
from pathlib import Path
from my_utils.my_xtb_utils import write_to_db, extract_energyxtb


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
        slurm_mem_per_cpu="500MB",
        timeout_min=8,
        slurm_partition="kemi1",
        slurm_array_parallelism=100,
    )

    jobs = executor.map_array(
        sc_function, population.molecules, [scoring_args for p in population.molecules]
    )

    results = [
        catch(job.result, handle=lambda e: (np.nan, None, None, None)) for job in jobs
    ]
    # catch submitit exceptions and return same output as scoring function
    # (np.nan, None) for (energy, geometry)
    if scoring_args["cleanup"]:
        shutil.rmtree("scoring_tmp")

    # Collect results in database

    if scoring_args["write_db"]:
        # Get traj paths for current gen
        p = Path(scoring_args["output_dir"])
        gen_no = f"{population.molecules[0].idx[0]}".zfill(3)

        trajs = sorted(p.rglob(f"{gen_no}*/*/*traj*"))
        logfiles = [p.parent / "xtbopt.log" for p in trajs]

        print("Printing optimized structures to database")
        try:
            # Write to database
            write_to_db(
                database_dir=scoring_args["database"],
                logfiles=logfiles,
                trajfile=trajs,
            )
        except Exception as e:
            print(f"Failed to write to database at {logfile}")
            print(e)

    return results


def catch(func, *args, handle=lambda e: e, **kwargs):
    """Helper function that takes the submitit result and returns an exception if
    no results can be retrieved"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)


### MolSimplify
# Submitit scoring functions related to molSimplify driver scripts


def slurm_scoring_molS(sc_function, scoring_args):

    executor = submitit.AutoExecutor(
        folder=Path(scoring_args["run_dir"]) / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"cycle",
        cpus_per_task=scoring_args["ncores"],
        slurm_mem_per_cpu="2GB",
        timeout_min=10,
        slurm_partition="kemi1",
        slurm_array_parallelism=2,
    )

    job = executor.submit(sc_function, **scoring_args)

    results = catch(job.result, handle=lambda e: None)

    # if scoring_args["cleanup"]:
    #    shutil.rmtree("scoring_tmp")

    return results


def slurm_scoring_molS_xtb(sc_function, scoring_args):

    executor = submitit.AutoExecutor(
        folder=scoring_args[-1] / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"xtb",
        cpus_per_task=scoring_args[-2],
        slurm_mem_per_cpu="2GB",
        timeout_min=10,
        slurm_partition="kemi1",
        slurm_array_parallelism=2,
    )

    job = executor.submit(sc_function, scoring_args)

    results = catch(job.result, handle=lambda e: (None, None))

    # if scoring_args["cleanup"]:
    #    shutil.rmtree("scoring_tmp")

    return results
