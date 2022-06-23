"""
Module that contains mol manipulations and various resuable functionality classes.
"""
import pickle
import random
import shutil
from collections import UserDict
import os, sys
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import re
import subprocess
from ase.io import read, write, Trajectory

from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
import concurrent.futures
import numpy as np

import pandas as pd
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import copy
from tabulate import tabulate
sys.path.insert(0, "../scoring")

from make_structures import create_prim_amine
from sa.neutralize import read_neutralizers
_neutralize_reactions = None

class DotDict(UserDict):
    """dot.notation access to dictionary attributes
    Currently not in use as it clashed with Multiprocessing-Pool pickling"""
    __getattr__ = UserDict.get

    __setattr__ = UserDict.__setitem__
    __delattr__ = UserDict.__delitem__


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
def draw_mol_with_highlights(mol, hit_ats, style=None):
    """Draw molecule in 3D with highlighted atoms.

    Parameters
    ----------
    mol : RDKit molecule
    hit_ats : tuple of tuples
        atoms to highlight, from RDKit's GetSubstructMatches
    style : dict, optional
        drawing style, see https://3dmol.csb.pitt.edu/doc/$3Dmol.GLViewer.html for some examples

    Returns
    -------
    py3Dmol viewer
    """
    v = py3Dmol.view()
    if style is None:
        style = {"stick": {"colorscheme": "grayCarbon", "linewidth": 0.1}}
    v.addModel(Chem.MolToMolBlock(mol), "mol")
    v.setStyle({"model": 0}, style)
    hit_ats = [x for tup in hit_ats for x in tup]
    for atom in hit_ats:
        p = mol.GetConformer().GetAtomPosition(atom)
        v.addSphere(
            {
                "center": {"x": p.x, "y": p.y, "z": p.z},
                "radius": 0.9,
                "color": "green",
                "alpha": 0.8,
            }
        )
    v.setBackgroundColor("white")
    v.zoomTo()
    return v


#### Example, use in Jupyter notebook:
# from rdkit.Chem import AllChem
# cyclosporine_smiles = "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C"
# cyclosporine = Chem.AddHs(Chem.MolFromSmiles(cyclosporine_smiles))
# AllChem.EmbedMolecule(cyclosporine)

# patt = Chem.MolFromSmarts('O[H]')
# hit_ats = cyclosporine.GetSubstructMatches(patt)
# draw_mol_with_highlights(cyclosporine, hit_ats)
def draw3d(
    mols,
    width=600,
    height=600,
    Hs=True,
    confId=-1,
    multipleConfs=False,
    atomlabel=False,
):
    try:
        p = py3Dmol.view(width=width, height=height)
        if type(mols) is not list:
            mols = [mols]
        for mol in mols:
            if multipleConfs:
                for conf in mol.GetConformers():
                    mb = Chem.MolToMolBlock(mol, confId=conf.GetId())
                    p.addModel(mb, "sdf")
            else:
                if type(mol) is str:
                    if os.path.splitext(mol)[-1] == ".xyz":
                        xyz_f = open(mol)
                        line = xyz_f.read()
                        xyz_f.close()
                        p.addModel(line, "xyz")
                    # elif os.path.splitext(mol)[-1] == '.out':
                    #     xyz_file = extract_optimized_structure(mol, return_mol=False)
                    #     xyz_f = open(xyz_file)
                    #     line = xyz_f.read()
                    #     xyz_f.close()
                    #     p.addModel(line,'xyz')
                else:
                    mb = Chem.MolToMolBlock(mol, confId=confId)
                    p.addModel(mb, "sdf")
        p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
        if atomlabel:
            p.addPropertyLabels("index")  # ,{'elem':'H'}
        p.zoomTo()
        p.update()
        # p.show()
    except:
        print("py3Dmol, RDKit, and IPython are required for this feature.")


def vis_trajectory(xyz_file, atomlabel=False):
    try:
        xyz_f = open(xyz_file)
        line = xyz_f.read()
        xyz_f.close()
        p = py3Dmol.view(width=400, height=400)
        p.addModelsAsFrames(line, "xyz")
        # p.setStyle({'stick':{}})
        p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
        if atomlabel:
            p.addPropertyLabels("index", {"elem": "H"})
        p.animate({"loop": "forward", "reps": 10})
        p.zoomTo()
        p.show()
    except:
        raise


def mol_from_xyz(xyz_file, charge=0):
    atoms, _, xyz_coordinates = read_xyz_file(xyz_file)
    mol = xyz2mol(atoms, xyz_coordinates, charge)
    return mol


def sdf2mol(sdf_file):
    mol = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)[0]
    return mol


def mols_from_smi_file(smi_file, n_mols=None):
    mols = []
    with open(smi_file) as _file:
        for i, line in enumerate(_file):
            mols.append(Chem.MolFromSmiles(line))
            if n_mols:
                if i == n_mols - 1:
                    break
    return mols


def mkdir(directory, overwrite=False):
    if os.path.exists(directory) and overwrite:
        shutil.rmtree(directory)
    os.mkdir(directory)


def hartree2kcalmol(hartree):
    return hartree * 627.5095


def hartree2kJmol(hartree):
    return hartree * 2625.50


@dataclass
class Individual:
    rdkit_mol: Chem.rdchem.Mol = field(repr=False, compare=False)
    original_mol: Chem.rdchem.Mol = field(
        default_factory=Chem.rdchem.Mol, repr=False, compare=False
    )
    rdkit_mol_sa: Chem.rdchem.Mol = field(default_factory=Chem.rdchem.Mol, repr=False, compare=False)
    cut_idx: int = field(default=None, repr=False, compare=False)
    idx: tuple = field(default=(None, None), repr=False, compare=False)
    smiles: str = field(init=False, compare=True, repr=True)
    smiles_sa: str = field(init=False, compare=True, repr=True)
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


@dataclass(order=True)
class Population:
    generation_num: int = field(default=None, init=True)
    molecules: List[Individual] = field(repr=False, default_factory=list)
    size: int = field(default=None, init=True)

    def __post_init__(self):
        self.size = len(self.molecules)

    def clean_mutated_survival_and_parents(self):
        for mol in self.molecules:
            mol.mutated = False
            mol.parentA_idx = None
            mol.parentB_idx = None
            mol.survival_idx = None

    def assign_idx(self):
        for i, molecule in enumerate(self.molecules):
            setattr(molecule, "idx", (self.generation_num, i))
        self.size = len(self.molecules)

    def sa_prep(self):
        for mol in self.molecules:
            output_ligand = AllChem.ReplaceSubstructs(
                mol.rdkit_mol,
                Chem.MolFromSmarts("[NX3;H2]"),
                Chem.MolFromSmarts("[H]"),
                replaceAll=True,
                replacementConnectionPoint=0,
            )[0]
            res = Chem.MolFromSmiles(Chem.MolToSmiles(output_ligand))
            mol.rdkit_mol_sa = res
            mol.smiles_sa = Chem.MolToSmiles(res)

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
            match = mol.rdkit_mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]"))
            mol.original_mol = mol.rdkit_mol

            # Create primary amine if it doesnt have once. Otherwise, pas the cut idx
            if not match:
                try:
                    output_ligand, cut_idx = create_prim_amine(mol.rdkit_mol)
                    # Handle if None is returned
                    if not output_ligand:
                        output_ligand = Chem.MolFromSmiles("CCCCCN")
                        cut_idx = [[1]]
                except Exception as e:
                    print("Could not create primary amine, setting methyl as ligand")
                    output_ligand = Chem.MolFromSmiles("CN")
                    cut_idx = [[1]]
                mol.rdkit_mol = output_ligand
                mol.cut_idx = cut_idx[0][0]
                mol.smiles = Chem.MolToSmiles(output_ligand)

            else:
                if supress_amines:
                    # Enable NH2 amine supressor.
                    if len(match) > 1:
                        # Replace other primary amines with hydrogen in the frag:
                        output_ligand = AllChem.ReplaceSubstructs(
                            mol.rdkit_mol,
                            Chem.MolFromSmarts("[NX3;H2]"),
                            Chem.MolFromSmarts("[H]"),
                            replacementConnectionPoint=0,
                        )[0]
                        clean_mol = Chem.MolFromSmiles(Chem.MolToSmiles(output_ligand))
                        mol.rdkit_mol = clean_mol
                        cut_idx = clean_mol.GetSubstructMatches(
                            Chem.MolFromSmarts("[NX3;H2]")
                        )
                        mol.cut_idx = cut_idx[0][0]
                        mol.smiles = Chem.MolToSmiles(clean_mol)
                    else:
                        cut_idx = random.choice(match)
                        mol.cut_idx = cut_idx[0]
                else:
                    cut_idx = random.choice(match)
                    mol.cut_idx = cut_idx[0]

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

    def print(self):
        table = []
        for individual in self.molecules:
            table.append(individual.list_of_props())
        if isinstance(self.generation_num, int):
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

    def pop2pd(self, columns=["score", "energy", "sa_score", "rdkit_mol"]):
        df = pd.DataFrame(
            list(map(list, zip(*[self.get(prop) for prop in columns]))),
            index=pd.MultiIndex.from_tuples(
                self.get("idx"), names=("generation", "individual")
            ),
        )
        df.columns = columns
        return df


@dataclass(order=True)
class Generation:
    generation_num: int = field(init=True)
    children: Population = field(default_factory=Population)
    survivors: Population = field(default_factory=Population)
    pre_children: Population = field(default_factory=Population)
    # a counter to count how ofter molecules got flagged during mutation/crossover by filter

    def __post_init__(self):
        if (
            self.generation_num != self.children.generation_num
            or self.generation_num != self.survivors.generation_num
        ):
            raise Warning(
                f"Generation {self.generation_num} has Children from generation {self.children.generation_num} and survivors from generation {self.survivors.generation_num}"
            )

    def save(self, directory=None, run_No=0):
        filename = os.path.join(directory, f"GA{run_No:02d}.pkl")
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def print(self, population="survivors", pass_text=None):
        if population == "survivors":
            population = self.survivors
        elif population == "children":
            population = self.children
        else:
            raise TypeError(f"{population} is not a valid option (survivors/children)")
        table = []
        for individual in population.molecules:
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
        for ind in self.children.molecules:
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
        population="survivors",
        columns=["score", "energy", "sa_score", "rdkit_mol"],
    ):
        if population == "survivors":
            population = self.survivors
        elif population == "children":
            population = self.children
        else:
            raise TypeError(f"{population} is not a valid option (survivors/children)")
        pd = population.pop2pd(columns=columns)
        return pd

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(generation_num={self.generation_num!r}, children_size={self.children.size}, survivors_size={self.survivors.size})"
        )


@dataclass(order=True)
class GA_run:
    num_generations: int = field(default=None, init=True)
    generations: List[Generation] = field(default_factory=list)

    def __post_init__(self):
        self.num_generations = len(self.generations)

    def append_gen(self, generation):
        self.generations += [generation]
        self.num_generations = len(self.generations)

    def get_ind(self, idx):
        return self.generations[idx[0]].survivors.molecules[idx[1]]

    def print(self, population="survivors"):
        for generation in self.generations:
            generation.print(population)

    def fails(self):
        nO_NaN = 0
        nO_9999 = 0
        for generation in self.generations:
            for ind in generation.children.molecules:
                tmp = ind.energy
                if np.isnan(tmp):
                    nO_NaN += 1
                elif tmp > 5000:
                    nO_9999 += 1
        return (nO_NaN, nO_9999)

    def ga2pd(
        self,
        population="survivors",
        columns=["score", "energy", "sa_score", "rdkit_mol"],
    ):
        df = pd.concat(
            [
                generation.gen2pd(population, columns=columns)
                for generation in self.generations
            ]
        )
        return df


def load_GA(pkl):
    with open(pkl, "rb") as rfp:
        ga = GA_run()
        while True:
            try:
                generation = pickle.load(rfp)
                ga.append_gen(generation)
            except EOFError:
                break
    return ga


def write_to_db(args):
    """

    Args:
        database_dir Path:
        structs List(xyz):

    Returns:

    """

    print("In write_db function")
    database_dir, trajfile = args

    logfile = trajfile.parent / "xtbopt.log"

    with connect(database_dir) as db:

        energies = extract_energyxtb(logfile)
        structs = read(trajfile, index=":")
        for i, (struct, energy) in enumerate(zip(structs, energies)):
            id = db.reserve(name=str(trajfile) + str(i))
            if id is None:
                continue
            struct.calc = SinglePointCalculator(struct, energy=energy)
            db.write(struct, id=id, name=str(trajfile))

    return


def write_to_traj(args):
    """

    Args:
        traj_dir Path:
        structs List(xyz):

    Returns:

    """

    print("In write_traj function")
    traj_dir, trajfile = args
    traj = Trajectory(traj_dir, mode="a")

    logfile = trajfile.parent / "xtbopt.log"
    energies = extract_energyxtb(logfile)
    struct = read(trajfile, index="-1")
    struct.calc = SinglePointCalculator(struct, energy=energies[-1])
    traj.write(struct)
    return


def db_write_driver(output_dir=None, workers=6):

    database_dir = "ase.traj"

    # Get traj paths for current gen
    p = Path(output_dir)

    trajs = p.rglob(f"0*/*/*traj*")

    # TODO paralellize the writing to database
    print("Printing optimized structures to database")
    try:

        args = [(database_dir, trajs) for i, trajs in enumerate(trajs)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(write_to_traj, args)
    except Exception as e:
        print(f"Failed to write to database at {logfile}")
        print(e)

    return


def extract_energyxtb(logfile=None):
    """
    Extracts xtb energies from xtb logfile using regex matching.

    Args:
        logfile (str): Specifies logfile to pull energy from

    Returns:
        energy (list[float]): List of floats containing the energy in each step
    """

    re_energy = re.compile("energy: (-\\d+\\.\\d+)")
    energy = []
    with logfile.open() as f:
        for line in f:
            if "energy" in line:
                energy.append(float(re_energy.search(line).groups()[0]))
    return energy


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


if __name__ == "__main__":
    db_write_driver("/home/magstr/generation_data/struct_test")
