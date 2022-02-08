import numpy as np
import shutil
import submitit


def slurm_scoring(sc_function, population, ids, cpus_per_task=4, cleanup=False):
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
        name=f"sc_g{ids[0][0]}",
        cpus_per_task=cpus_per_task,
        slurm_mem_per_cpu="1GB",
        timeout_min=30,
        slurm_partition="kemi1",
        slurm_array_parallelism=100,
    )
    args = [cpus_per_task for p in population]
    jobs = executor.map_array(sc_function, population, ids, args)

    results = [
        catch(job.result, handle=lambda e: (np.nan, None)) for job in jobs
    ]  # catch submitit exceptions and return same output as scoring function (np.nan, None) for (energy, geometry)
    if cleanup:
        shutil.rmtree("scoring_tmp")
    return results


def catch(func, *args, handle=lambda e: e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)


