import sys
import os
from rdkit import Chem

from pathlib import Path
import numpy as np
import random
from tabulate import tabulate

from typing import List

sys.path.append(os.path.dirname(__file__))

import crossover as co
import scoring_functions as sc
import GB_GA as ga
import filters
from sa import sa_target_score_clipped, neutralize_molecules
from descriptors import number_of_rotatable_bonds_target_clipped
from catalyst.utils import Individual
from catalyst import ts_scoring


def reweigh_scores_by_sa(
    population: List[Chem.Mol], scores: List[float]
) -> List[float]:
    """Reweighs scores with synthetic accessibility score
    :param population: list of RDKit molecules to be re-weighted
    :param scores: list of docking scores
    :return: list of re-weighted docking scores
    """
    sa_scores = [sa_target_score_clipped(p) for p in population]
    return sa_scores, [
        ns * sa for ns, sa in zip(scores, sa_scores)
    ]  # rescale scores and force list type


def reweigh_scores_by_number_of_rotatable_bonds_target(
    population: List[Chem.Mol],
    scores: List[float],
    nrb_target=5,  # true mean is 4.510575601574693
    nrb_standard_deviation=2,
):  # true std is 1.5511682274280625
    """Reweighs docking scores by number of rotatable bonds.
    For some molecules we want a maximum of number of rotatable bonds (typically 5) but
    we want to keep some molecules with a larger number around for possible mating.
    The default parameters keeps all molecules with 5 rotatable bonds and roughly 40 %
    of molecules with 6 rotatable bonds.
    :param population:
    :param scores:
    :param molecule_options:
    :return:
    """
    number_of_rotatable_target_scores = [
        number_of_rotatable_bonds_target_clipped(p, nrb_target, nrb_standard_deviation)
        for p in population
    ]
    return [
        ns * lts for ns, lts in zip(scores, number_of_rotatable_target_scores)
    ]  # rescale scores and force list type


def print_results(population, fitness, generation):
    print(f"\nGeneration {generation+1}", flush=True)
    print(
        tabulate(
            [
                [ind.idx, fit, ind.score, ind.energy, ind.sa_score, ind.smiles]
                for ind, fit in zip(population, fitness)
            ],
            headers=[
                "idx",
                "normalized fitness",
                "score",
                "energy",
                "sa score",
                "smiles",
            ],
        ),
        flush=True,
    )


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
        seed,
        minimization,
        selection_method,
        selection_pressure,
        molecule_filters,
        path,
    ) = args

    np.random.seed(seed)
    random.seed(seed)

    molecules = ga.make_initial_population(population_size, file_name, random=False)

    ids = [(0, i) for i in range(len(molecules))]
    results = sc.slurm_scoring(scoring_function, molecules, ids)
    energies = [res[0] for res in results]
    geometries = [res[1] for res in results]

    prescores = [energy - 100 for energy in energies]
    sa_scores, scores = reweigh_scores_by_sa(neutralize_molecules(molecules), prescores)
    scores = reweigh_scores_by_number_of_rotatable_bonds_target(molecules, scores)

    population = [
        Individual(
            idx=idx,
            rdkit_mol=mol,
            score=score,
            energy=energy,
            sa_score=sa_score,
            structure=structure,
        )
        for idx, mol, score, energy, sa_score, structure in zip(
            ids, molecules, scores, energies, sa_scores, geometries
        )
    ]
    population = ga.sanitize(population, population_size, False)

    fitness = ga.calculate_fitness(
        [ind.score for ind in population],
        minimization,
        selection_method,
        selection_pressure,
    )
    fitness = ga.calculate_normalized_fitness(fitness)

    print_results(population, fitness, -1)

    for generation in range(generations):
        mating_pool = ga.make_mating_pool(population, fitness, mating_pool_size)

        new_population = ga.reproduce(
            mating_pool,
            population_size,
            mutation_rate,
            molecule_filters,
            generation + 1,
        )

        new_resuls = sc.slurm_scoring(
            scoring_function,
            [ind.rdkit_mol for ind in new_population],
            [ind.idx for ind in new_population],
        )

        new_energies = [res[0] for res in new_resuls]
        new_geometries = [res[1] for res in new_resuls]

        new_prescores = [energy - 100 for energy in new_energies]
        new_sa_scores, new_scores = reweigh_scores_by_sa(
            neutralize_molecules([ind.rdkit_mol for ind in new_population]),
            new_prescores,
        )
        new_scores = reweigh_scores_by_number_of_rotatable_bonds_target(
            molecules, new_scores
        )

        for ind, score, energy, sa_score, structure in zip(
            new_population,
            new_scores,
            new_energies,
            new_sa_scores,
            new_geometries,
        ):
            ind.score = score    (
        n_confs,
        randomseed,
        timing_logger,
        warning_logger,
        directory,
        cpus_per_molecule,
    ) = args_list
            ind.energy = energy
            ind.sa_score = sa_score
            ind.structure = structure

        population = ga.sanitize(
            population + new_population, population_size, prune_population
        )

        fitness = ga.calculate_fitness(
            [ind.score for ind in population],
            minimization,
            selection_method,
            selection_pressure,
        )
        fitness = ga.calculate_normalized_fitness(fitness)

        print_results(population, fitness, generation)


### ----------------------------------------------------------------------------------


if __name__ == "__main__":

    package_directory = Path(__file__).parent.resolve()

    co.average_size = 8.0  # 14 24.022840038202613
    co.size_stdev = 4.0  # 8 4.230907997270275
    population_size = 100
    # file_name = package_directory / "ZINC_amines.smi"
    scoring_function = ts_scoring
    generations = 75
    mating_pool_size = population_size
    mutation_rate = 0.50
    scoring_args = None
    prune_population = True
    seed = 101
    minimization = True
    selection_method = "rank"
    selection_pressure = 1.5
    molecule_filters = filters.get_molecule_filters(
        ["Julius"], package_directory / "filters/alert_collection.csv"
    )
    file_name = sys.argv[-1]

    path = "."

    args = [
        population_size,
        file_name,
        scoring_function,
        generations,
        mating_pool_size,
        mutation_rate,
        scoring_args,
        prune_population,
        seed,
        minimization,
        selection_method,
        selection_pressure,
        molecule_filters,
        path,
    ]

    GA(args)
