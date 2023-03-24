# -*- coding: utf-8 -*-
"""Module that contains submitit functionality to submitit scoring function."""

import shutil
from pathlib import Path

import numpy as np
import submitit

from scoring.scoring import scoring_submitter
from utils.utils import catch


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
    mem_per_cpu = (scoring_args["memory"] * 1000) // scoring_args["cpus_per_task"]
    executor.update_parameters(
        name=f"conformer_search",
        cpus_per_task=scoring_args["cpus_per_task"],
        slurm_mem_per_cpu=f"{mem_per_cpu}MB",
        timeout_min=scoring_args["timeout"],
        slurm_partition=scoring_args["partition"],
        slurm_array_parallelism=20,
    )

    jobs = executor.map_array(
        scoring_submitter,
        conformers.molecules,
        [scoring_args for c in conformers.molecules],
    )

    # Get the jobs results. Assign None variables if an error is returned for the given molecule
    # (np.nan, None, None, None) for (energy, geometry1, geometry2, minidx)
    results = [
        catch(
            job.result,
            handle=lambda e: (
                None,
                None,
                {"energy1": None, "energy2": None, "score": np.nan},
            ),
        )
        for job in jobs
    ]

    if scoring_args["cleanup"]:
        shutil.rmtree(Path(scoring_args["output_dir"]) / "scoring_tmp")

    return results


### MolSimplify
# Submitit scoring functions related to molSimplify driver scripts


def slurm_molS(sc_function, scoring_args):
    """To submit create_cycle_MS to the commandline and create all Mo
    intermediates with a given ligand.

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
