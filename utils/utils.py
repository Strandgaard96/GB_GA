"""Module that contains mol manipulations and various reusable
functionality."""
import copy
import os
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
    """Filter out higher energy conformers based on energy cutoff.

    Args:
        confs: Sequnce of conformers objects.
        energies (List): List of conformer energies
        optimized_mol (Chem.Mol): Optimized mol object
        scoring_args: Scoring function arg dict

    Returns:
        energies: Filtered energies
        new_mol: Mol object with filtered conformers
    """

    mask = energies < (energies.min() + scoring_args["energy_cutoff"])
    print(mask, energies)
    confs = list(np.array(confs)[mask])
    new_mol = copy.deepcopy(optimized_mol)
    new_mol.RemoveAllConformers()
    for c in confs:
        new_mol.AddConformer(c, assignId=True)
    energies = energies[mask]

    return energies, new_mol


def shell(args, shell=False):
    """Subprocess handler function where output is stored in files.

    Args:
        cmd (str): String to pass to bash shell
        shell (bool): Specifies whether run as bash shell or not

    Returns:
        output (str): Program output
        err (str): Possible error messages
    """
    cmd, key = args
    print(f"String passed to shell: {cmd}")
    if shell:
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    else:
        cmd = cmd.split()
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    output, err = p.communicate()

    with open(f"{key}job.out", "w") as f:
        f.write(output)
    with open(f"{key}err.out", "w") as f:
        f.write(err)

    return output, err


class cd:
    """Context manager for changing the current working directory dynamically.

    # See: https://book.pythontips.com/en/latest/context_managers.html
    """

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


def read_file(file_name):
    """Read smiles from file and return mol generator."""
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            yield Chem.MolFromSmiles(smiles)


def catch(func, *args, handle=lambda e: e, **kwargs):
    """Helper function that takes the submitit result and returns an exception
    if no results can be retrieved."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)
