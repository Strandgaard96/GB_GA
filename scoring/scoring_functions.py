# -*- coding: utf-8 -*-
"""
Module that contains submitit scoring functions. These submit multiple
scoring calculation jobs.
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import submitit
from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS

from my_utils.xtb_utils import extract_energyxtb, write_to_db
from scoring.scoring import (
    rdkit_embed_scoring,
    rdkit_embed_scoring_NH3plustoNH3,
    rdkit_embed_scoring_NH3toN2,
)

funcs = {
    "rdkit_embed_scoring": rdkit_embed_scoring,
    "rdkit_embed_scoring_NH3toN2": rdkit_embed_scoring_NH3toN2,
    "rdkit_embed_scoring_NH3plustoNH3": rdkit_embed_scoring_NH3plustoNH3,
}


def slurm_scoring(sc_function, population, scoring_args):
    """Evaluates a scoring function for population on SLURM cluster
    Args:
        sc_function (function): Scoring function to use for each molecule
        population List(Individual): List of molecules objects to score
        scoring_args (dict): Relevant scoring args for submitit or XTB
    Returns:
        results List(tuple): List of tuples containing result for each molecule
    """
    executor = submitit.AutoExecutor(
        folder=Path(scoring_args["output_dir"]) / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"sc_g{population.molecules[0].idx[0]}",
        cpus_per_task=scoring_args["cpus_per_task"],
        slurm_mem_per_cpu="500MB",
        timeout_min=scoring_args["timeout"],
        slurm_partition="kemi1",
        slurm_array_parallelism=100,
    )

    jobs = executor.map_array(
        sc_function, population.molecules, [scoring_args for p in population.molecules]
    )

    # Get the jobs results. Assign None variables if an error is returned for the given molecule
    # (np.nan, None, None, None) for (energy, geometry1, geometry2, minidx)
    results = [
        catch(job.result, handle=lambda e: (np.nan, None, None, None)) for job in jobs
    ]

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


def slurm_scoring_conformers(conformers, scoring_args):
    """Evaluates a scoring function for population on SLURM cluster
    Args:
        conformers List(Individual): List of molecules objects to do conformers search for
        scoring_args (dict): Relevant scoring args for submitit or XTB
    Returns:
        results List(tuple): List of tuples containing result for each molecule
    """
    executor = submitit.AutoExecutor(
        folder=Path(scoring_args["output_dir"]) / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    executor.update_parameters(
        name=f"conformer_search",
        cpus_per_task=scoring_args["cpus_per_task"],
        slurm_mem_per_cpu="500MB",
        timeout_min=scoring_args["timeout"],
        slurm_partition=scoring_args["partition"],
        slurm_array_parallelism=100,
    )

    list_scoring = []
    for mol in conformers.molecules:
        tmp = scoring_args.copy()
        tmp["output_dir"] = scoring_args["output_dir"]/mol.scoring_function
        list_scoring.append(tmp)

    jobs = [
        executor.submit(funcs[mol.scoring_function], mol, scoring_args)
        for mol, scoring_args in zip(conformers.molecules, list_scoring)
    ]

    # Get the jobs results. Assign None variables if an error is returned for the given molecule
    # (np.nan, None, None, None) for (energy, geometry1, geometry2, minidx)
    results = [
        catch(job.result, handle=lambda e: (np.nan, None, None, None)) for job in jobs
    ]

    if scoring_args["cleanup"]:
        shutil.rmtree("scoring_tmp")

    # Collect results in database
    if scoring_args["write_db"]:
        # Get traj paths for current gen
        p = Path(scoring_args["output_dir"])
        gen_no = f"{conformers.molecules[0].idx[0]}".zfill(3)

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


def slurm_molS(sc_function, scoring_args):
    """
    To submit create_cycle_MS to the commandline and create all
    Mo intermediates with a given ligand.

    Args:
        sc_function (func): molS driver function
        scoring_args (dict): Relevant scoring args

    Returns:
        results List(tuples): Resulst of molS output, not used.

    """
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

    return results


def slurm_molS_xtb(sc_function, scoring_args):
    """
    Function is outdated and should be updated. Was used to submit all
    intermediates to xtb calcs after molS was used to create them.
    """
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

    return results
