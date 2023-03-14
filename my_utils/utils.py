"""Module that contains mol manipulations and various reusable
functionality."""
import copy
import os
import shutil
import subprocess
import sys

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory, read
from rdkit import Chem

# Needed for debugging
sys.path.insert(0, "../scoring")


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


def write_to_traj(args):
    """Write xtblog files to traj file."""

    traj_dir, trajfile = args
    traj = Trajectory(traj_dir, mode="a")

    logfile = trajfile.parent / "xtbopt.log"
    energies = extract_energyxtb(logfile)
    struct = read(trajfile, index="-1")
    struct.calc = SinglePointCalculator(struct, energy=energies[-1])
    traj.write(struct)

    return


def get_git_revision_short_hash() -> str:
    """Get the git hash of current commit for each GA run."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def energy_filter(confs, energies, optimized_mol, scoring_args):

    mask = energies < (energies.min() + scoring_args["energy_cutoff"])
    print(mask, energies)
    confs = list(np.array(confs)[mask])
    new_mol = copy.deepcopy(optimized_mol)
    new_mol.RemoveAllConformers()
    for c in confs:
        new_mol.AddConformer(c, assignId=True)
    energies = energies[mask]

    return energies, new_mol
