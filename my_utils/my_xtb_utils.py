from rdkit import Chem
from rdkit.Chem import AllChem
from xyz2mol import read_xyz_file, xyz2mol
from .auto import shell

import os
import random
import shutil
import string
import subprocess
import logging
import json


# %%
def write_xtb_input_files(fragment, name, destination="."):
    number_of_atoms = fragment.GetNumAtoms()
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
    conformers = fragment.GetConformers()
    file_paths = []
    for i, conf in enumerate(fragment.GetConformers()):
        conf_path = os.path.join(destination, f"conf{i:03d}")
        os.mkdir(conf_path)
        file_name = f"{name}{i:03d}.xyz"
        file_path = os.path.join(conf_path, file_name)
        with open(file_path, "w") as _file:
            _file.write(str(number_of_atoms) + "\n")
            _file.write(f"{Chem.MolToSmiles(fragment)}\n")
            for atom, symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                _file.write(line)
        file_paths.append(file_path)
    return file_paths

def check_xtb_input():
    # Perform various checks. The exceptions here should be caught in driver script!
    if isinstance(mol, Chem.rdchem.Mol):
        # Check for implicit hydrogens
        if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
            raise Exception("Implicit Hydrogens")
        # Check for embedding
        conformers = mol.GetConformers()
        if not conformers:
            raise Exception("Mol is not embedded")
        elif not conformers[-1].Is3D():
            raise Exception("Conformer is not 3D")
        true_charge = Chem.GetFormalCharge(mol)

        # Write mol objects to xyz files for xtb input
        xyz_files = write_xtb_input_files(mol, "xtbmol", destination=dest)
    else:
        raise Exception("Not mol object")


def run_xtb(structure, type, method ,charge, spin, numThreads=None,**kwargs):

    # Get numbre of cores available.
    # Careful with this when parallellizing
    if not numThreads:
        numThreads = os.cpu_count()

    # Set environmental variables
    os.environ["OMP_NUM_THREADS"] = f"{numThreads},1"
    os.environ["MKL_NUM_THREADS"] = f"{numThreads}"
    os.environ["OMP_STACKSIZE"] = "4G"

    # Default input string
    cmd = f"xtb {structure} --{type} --{method} --chrg {charge} --uhf {spin}"

    # Add extra xtb settings if provided
    if kwargs:
        for key, value in kwargs.items():
            cmd += f" --{key} {value}"

    # Run the shell command
    print(f'Running xtb with input string {cmd}')
    out, err = shell(
        cmd,
        shell=True,
    )
    with open('job.out', 'w') as f:
        f.write(out)
    with open('err.out', 'w') as f:
        f.write(err)
    #print(out, file=open("job.out", "a"))

    # Post processing
    # Insert pos processing

    return out, err


def extract_energyxtb(logfile=None):
    '''
    Extracts xtb energies from xtb logfile using regex matching.

    Args:
        logfile (str): Specifies logfile to pull energy from

    Returns:
        energy (list[float]): List of floats containing the energy in each step
    '''

    re_energy = re.compile("energy: (-\\d+\\.\\d+)")
    energy = []
    with logfile.open() as f:
        for line in f:
            if "energy" in line:
                energy.append(float(re_energy.search(line).groups()[0]))
    return energy
