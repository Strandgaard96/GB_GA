"""Module containing classes used in the GA and conformer searches."""

import os
import pickle
from dataclasses import dataclass, field
from inspect import getmembers, isfunction
from pathlib import Path

import pandas as pd
import submitit
from rdkit import Chem
from tabulate import tabulate

from scoring import scoring_functions as sc
from utils.utils import catch, read_file

functions_dict = dict(getmembers(sc, isfunction))


@dataclass(eq=True)
class Individual:
    """Dataclass for storing data for each molecule.

    The central objects of the GA. The moles themselves plus various
    attributes and debugging fields are set.

    Attributes:
        rdkit_mol: The rdkit mol object
        original_mol: The mol object at the start of a generation.
        rdkit_mol_sa: Mol object where the primary amine is replaced with a hydrogen.
         Used for the SA score.
        cut_idx: The index of the primary amine that denotes the attachment point.
        idx: The generation idx of the molecule.
        smiles: SMILES representation of molecule.
        smiles_sa: SMILES representation of the molecule with primary amine replaced with
        hydrogen for SA score.
        score: GA score for the molecule.
        normalized_fitness: Normalized score value for the current population.
        energy: Reaction energy for the scoring step.
        sa_score: Synthetic accessibility score.
    """

    rdkit_mol: Chem.rdchem.Mol = field(repr=False, compare=True)
    original_mol: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    rdkit_mol_sa: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    cut_idx: int = field(default=None, repr=False, compare=False)
    idx: tuple = field(default=(None, None), repr=False, compare=False)
    smiles: str = field(init=False, compare=True, repr=True)
    smiles_sa: str = field(init=False, compare=False, repr=False)
    score: float = field(default=None, repr=False, compare=False)
    normalized_fitness: float = field(default=None, repr=False, compare=False)
    energy: float = field(default=None, repr=False, compare=False)
    sa_score: float = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.smiles = Chem.MolToSmiles(self.rdkit_mol)

    def get(self, prop):
        """Get property from an individual."""
        prop = getattr(self, prop)
        return prop

    def save(self, directory="."):
        """Dump ind object into file."""
        filename = os.path.join(directory, f"ind.pkl")
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def get_props(self):
        return vars(self)

    # To enable the set method and comparing the smiles of molecules.
    def __hash__(self) -> int:
        return hash(self.smiles)


class OutputHandler:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def save(molecules, directory=None, name=None):
        """Save instance to file for later retrieval."""
        filename = os.path.join(directory, name)
        with open(filename, "ab+") as output:
            pickle.dump(molecules, output, pickle.HIGHEST_PROTOCOL)

    def write_out(self, molecules, name=None):
        self.molecules = molecules
        with open(self.args["output_dir"] / name, "w") as f:
            f.write(self.print(pass_text=True) + "\n")
            f.write(self.print_fails())

    def print(self, pass_text=None):
        """Print nice table of population attributes."""
        table = []
        relevant_props = [
            "smiles",
            "idx",
            "normalized_fitness",
            "energy",
            "score",
            "sa_score",
        ]
        for individual in self.molecules:
            props = individual.get_props()
            property_row = [props[key] for key in relevant_props]
            table.append(property_row)
        txt = tabulate(table, headers=relevant_props)
        print(txt)
        if pass_text:
            return txt

    def print_fails(self):
        """Log how many calcs in population failed."""
        nO_NaN = 0

        [nO_NaN + 1 for x in self.molecules if not x.energy]

        table = [[nO_NaN]]
        txt = tabulate(
            table,
            headers=["Number of NaNs"],
        )
        print(txt)
        return txt

    def pop2pd(
        self,
        columns=["cut_idx", "score", "energy", "sa_score", "smiles"],
    ):
        """Get dataframe of population."""
        df = pd.DataFrame(
            list(map(list, zip(*[self.get(prop) for prop in columns]))),
            index=pd.MultiIndex.from_tuples(
                self.get("idx"), names=("generation", "individual")
            ),
        )
        df.columns = columns
        return df

    def pop2pd_dft(self):
        columns = [
            "smiles",
            "idx",
            "cut_idx",
            "score",
            "energy",
            "dft_singlepoint_conf",
            "min_confs",
        ]
        """Get dataframe of population."""
        df = pd.DataFrame(list(map(list, zip(*[self.get(prop) for prop in columns]))))
        df.columns = columns
        return df


class SubmitIt:
    """Score a population of molecules."""

    def __init__(self, args):
        self.args = args

    def score_population(self, molecules):
        """Run scoring."""
        # Setup submitit executor
        self._initialize_submitit(name=f"sc_g{molecules[0].idx[0]}")

        jobs = self.executor.map_array(self.selected_function, molecules)

        # Get the jobs results.
        scored_molecules = [
            catch(job.result, handle=lambda e: self.molecules[i])
            for i, job in enumerate(jobs)
        ]

        return scored_molecules

    def _initialize_submitit(self, name=None):

        self.executor = submitit.AutoExecutor(
            folder=Path(self.args["output_dir"]) / "scoring_tmp",
            slurm_max_num_timeout=0,
            cluster="debug",
        )
        self.executor.update_parameters(
            name=name,
            cpus_per_task=self.args["cpus_per_task"],
            slurm_mem_per_cpu=self.args["mem_per_cpu"],
            timeout_min=self.args["timeout"],
            slurm_partition=self.args["partition"],
            slurm_array_parallelism=100,
        )


class ScoringFunction(SubmitIt):
    def scoring_submitter(self, population):

        self.selected_function = functions_dict[self.args["scoring_function"]]
        scored_molecules = self.score_population(population)

        return scored_molecules


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.filename = args["filename"]

    def load_data(self):
        """Create starting population from csv file."""

        # Get mol generator from file
        mol_generator = read_file(self.filename)
        initial_population = []

        for i in range(self.args["population_size"]):

            candidate_match = False
            while not candidate_match:
                mol = next(mol_generator)
                # Match amines, not bound to amines in rings or other amines
                candidate_match = mol.GetSubstructMatches(Chem.MolFromSmarts("*"))

            # TODO process mol

            initial_population.append(Individual(rdkit_mol=mol))

        return initial_population

    def load_debug(self):

        initial_population = []

        # Smiles with primary amines and corresponding cut idx
        smiles = ["CCN", "NC1CCC1", "CCN", "CCN"]
        idx = [2, 0, 2, 2]

        for i in range(len(smiles)):
            ligand = Chem.MolFromSmiles(smiles[i])
            cut_idx = [[idx[i]]]
            initial_population.append(Individual(ligand, cut_idx=cut_idx[0][0]))

        return initial_population
