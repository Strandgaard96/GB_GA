"""
Module that contains mol manipulations and various resuable functionality classes.
"""
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import Trajectory, read
from rdkit import Chem

# Needed for when debugging
sys.path.insert(0, "../scoring")

_neutralize_reactions = None


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
    """Write xtblog files to traj file"""

    traj_dir, trajfile = args
    traj = Trajectory(traj_dir, mode="a")

    logfile = trajfile.parent / "xtbopt.log"
    energies = extract_energyxtb(logfile)
    struct = read(trajfile, index="-1")
    struct.calc = SinglePointCalculator(struct, energy=energies[-1])
    traj.write(struct)

    return


def db_write_driver(output_dir=None, workers=6):
    """Paralellize writing to db, not very robust"""

    database_dir = "ase.traj"

    # Get traj paths for current gen
    p = Path(output_dir)

    trajs = p.rglob(f"0*/*/*traj*")

    print("Printing optimized structures to database")
    try:

        args = [(database_dir, trajs) for i, trajs in enumerate(trajs)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(write_to_traj, args)
    except Exception as e:
        print(f"Failed to write to database at {logfile}")
        print(e)

    return


def get_git_revision_short_hash() -> str:
    """Get the git hash of current commit for each GA run"""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


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


if __name__ == "__main__":
    db_write_driver("/home/magstr/generation_data/struct_test")
