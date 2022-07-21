import copy
import os
import pickle
import random
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from tabulate import tabulate

from sa.neutralize import read_neutralizers
from scoring.make_structures import atom_remover, create_prim_amine_revised

_neutralize_reactions = None

class cd:
    """Context manager for changing the current working directory dynamically.
    # See: https://book.pythontips.com/en/latest/context_managers.html"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        # Print traceback if anything happens
        if traceback:
            print(sys.exc_info())
        os.chdir(self.savedPath)


@dataclass
class Individual:
    rdkit_mol: Chem.rdchem.Mol = field(repr=False, compare=False)
    original_mol: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    rdkit_mol_sa: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    cut_idx: int = field(default=None, repr=False, compare=False)
    idx: tuple = field(default=(None, None), repr=False, compare=False)
    smiles: str = field(init=False, compare=True, repr=True)
    smiles_sa: str = field(init=False, compare=True, repr=False)
    score: float = field(default=None, repr=False, compare=False)
    normalized_fitness: float = field(default=None, repr=False, compare=False)
    energy: float = field(default=None, repr=False, compare=False)
    sa_score: float = field(default=None, repr=False, compare=False)
    structure: tuple = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        self.smiles = Chem.MolToSmiles(self.rdkit_mol)

    def list_of_props(self):
        return [
            self.idx,
            self.normalized_fitness,
            self.score,
            self.energy,
            self.sa_score,
            self.smiles,
        ]

    def save(self, directory="."):
        filename = os.path.join(directory, f"ind.pkl")
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


@dataclass(order=True)
class Generation:
    generation_num: int = field(init=True, default=None)
    molecules: List[Individual] = field(repr=True, default_factory=list)
    new_molecules: List[Individual] = field(repr=False, default_factory=list)
    size: int = field(default=None, init=True, repr=True)

    def __post_init__(self):
        self.size = len(self.molecules)

    def __repr__(self):
        return (
            f"" f"(generation_num={self.generation_num!r}, molecules_size={self.size})"
        )

    def assign_idx(self):
        for i, molecule in enumerate(self.molecules):
            setattr(molecule, "idx", (self.generation_num, i))
        self.size = len(self.molecules)

    def save(self, directory=None, run_No=0):
        filename = os.path.join(directory, f"GA{run_No:02d}.pkl")
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_debug(self, directory=None, run_No=0):
        filename = os.path.join(directory, f"GA{run_No:02d}_debug.pkl")
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_debug2(self, directory=None, run_No=0):
        filename = os.path.join(directory, f"GA{run_No:02d}_debug2.pkl")
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def get(self, prop):
        properties = []
        for molecule in self.molecules:
            properties.append(getattr(molecule, prop))
        return properties

    def setprop(self, prop, list_of_values):
        for molecule, value in zip(self.molecules, list_of_values):
            setattr(molecule, prop, value)

    def appendprop(self, prop, list_of_values):
        for molecule, value in zip(self.molecules, list_of_values):
            if value:
                getattr(molecule, prop).append(value)

    def sortby(self, prop, reverse=True):
        if reverse:
            self.molecules.sort(
                key=lambda x: float("inf") if np.isnan(x.score) else x.score,
                reverse=reverse,
            )
        else:
            self.molecules.sort(
                key=lambda x: float("inf") if np.isnan(x.score) else x.score,
                reverse=reverse,
            )

    def prune(self, population_size):
        self.sortby("score", reverse=False)
        self.molecules = self.molecules[:population_size]
        self.size = len(self.molecules)

    def print(self, population="molecules", pass_text=None):
        table = []
        if population == "molecules":
            population = self.molecules
        elif population == "new_molecules":
            population = self.new_molecules
        for individual in population:
            table.append(individual.list_of_props())
        print(f"\nGeneration {self.generation_num:02d}")
        print(
            tabulate(
                table,
                headers=[
                    "idx",
                    "normalized_fitness",
                    "score",
                    "energy",
                    "sa_score",
                    "smiles",
                ],
            )
        )
        if pass_text:
            txt = tabulate(
                table,
                headers=[
                    "idx",
                    "normalized_fitness",
                    "score",
                    "energy",
                    "sa_score",
                    "smiles",
                ],
            )
            return txt

    def summary(self):
        nO_NaN = 0
        nO_9999 = 0
        for ind in self.new_molecules:
            tmp = ind.energy
            if np.isnan(tmp):
                nO_NaN += 1
            elif tmp > 5000:
                nO_9999 += 1
        table = [[nO_NaN, nO_9999, nO_NaN + nO_9999]]
        txt = tabulate(
            table,
            headers=["Number of NaNs", "Number of high energies", "Total"],
        )
        return txt

    def gen2pd(
        self,
        columns=["score", "energy", "sa_score", "rdkit_mol"],
    ):
        df = pd.DataFrame(
            list(map(list, zip(*[self.get(prop) for prop in columns]))),
            index=pd.MultiIndex.from_tuples(
                self.get("idx"), names=("generation", "individual")
            ),
        )
        df.columns = columns
        return df

    def update_property_cache(self):
        for mol in self.molecules:

            # Done to prevent ringinfo error
            Chem.GetSymmSSSR(mol.rdkit_mol)
            mol.rdkit_mol.UpdatePropertyCache()

    def sa_prep(self):
        for mol in self.molecules:
            prim_match = Chem.MolFromSmarts("[NX3;H2]")

            # Substructure match the NH3
            ms = [x for x in atom_remover(mol.rdkit_mol, pattern=prim_match)]
            removed_mol = random.choice(ms)
            prim_amine_index = removed_mol.GetSubstructMatches(
                Chem.MolFromSmarts("[NX3;H2]")
            )
            mol.rdkit_mol_sa = removed_mol
            mol.smiles_sa = Chem.MolToSmiles(removed_mol)

        global _neutralize_reactions
        if _neutralize_reactions is None:
            _neutralize_reactions = read_neutralizers()

        neutral_molecules = []
        for ind in self.molecules:
            c_mol = ind.rdkit_mol_sa
            mol = copy.deepcopy(c_mol)
            mol.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(mol)
            assert mol is not None
            for reactant_mol, product_mol in _neutralize_reactions:
                while mol.HasSubstructMatch(reactant_mol):
                    rms = Chem.ReplaceSubstructs(mol, reactant_mol, product_mol)
                    if rms[0] is not None:
                        mol = rms[0]
            mol.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(mol)
            ind.neutral_rdkit_mol = mol

    def set_sa(self, sa_scores):
        for individual, sa_score in zip(self.molecules, sa_scores):
            individual.sa_score = sa_score
            if individual.score > 5000:
                continue
            else:
                individual.score = sa_score * individual.pre_score

    def modify_population(self, supress_amines=False):
        for mol in self.molecules:
            # Check for primary amine
            match = mol.rdkit_mol.GetSubstructMatches(
                Chem.MolFromSmarts("[NX3;H2;!$(*n);!$(*N)]")
            )
            mol.original_mol = mol.rdkit_mol

            # Create primary amine if it doesnt have once. Otherwise, pas the cut idx
            if not match:
                try:
                    output_ligand, cut_idx = create_prim_amine_revised(mol.rdkit_mol)
                    # output_ligand, cut_idx = create_prim_amine(mol.rdkit_mol)
                    # Handle if None is returned
                    if not output_ligand:
                        output_ligand = Chem.MolFromSmiles("CCCCCN")
                        cut_idx = [[1]]
                except Exception as e:
                    print("Could not create primary amine, setting methyl as ligand")
                    output_ligand = Chem.MolFromSmiles("CN")
                    cut_idx = [[1]]
                mol.rdkit_mol = Chem.MolFromSmiles(Chem.MolToSmiles(output_ligand))
                mol.cut_idx = cut_idx[0][0]
                mol.smiles = Chem.MolToSmiles(
                    Chem.MolFromSmiles(Chem.MolToSmiles(output_ligand))
                )

            else:
                cut_idx = random.choice(match)
                mol.cut_idx = cut_idx[0]

                if supress_amines:

                    # Check for N-N bound amines
                    nn_match = mol.rdkit_mol.GetSubstructMatches(
                        Chem.MolFromSmarts("[NX3;H2;$(*N),$(*n)]")
                    )

                    # Enable NH2 amine supressor.
                    if len(match) > 1:

                        # Substructure match the NH3
                        prim_match = Chem.MolFromSmarts("[NX3;H2]")
                        # Substructure match the NH3
                        ms = [
                            x for x in atom_remover(mol.rdkit_mol, pattern=prim_match)
                        ]
                        removed_mol = random.choice(ms)
                        prim_amine_index = removed_mol.GetSubstructMatches(
                            Chem.MolFromSmarts("[NX3;H2]")
                        )
                        mol.rdkit_mol = removed_mol
                        mol.cut_idx = prim_amine_index[0][0]
                        mol.smiles = Chem.MolToSmiles(removed_mol)

                    elif nn_match:
                        # Replace other primary amines with hydrogen in the frag:
                        prim_match = Chem.MolFromSmarts("[NX3;H2;$(*N),$(*n)]")

                        rm = AllChem.DeleteSubstructs(mol.rdkit_mol, prim_match)

                        prim_amine_index = rm.GetSubstructMatches(
                            Chem.MolFromSmarts("[NX3;H2]")
                        )
                        mol.rdkit_mol = rm
                        mol.cut_idx = prim_amine_index[0][0]
                        mol.smiles = Chem.MolToSmiles(rm)
