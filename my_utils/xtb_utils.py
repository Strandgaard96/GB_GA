import concurrent.futures
import json
import logging
import os
import random
import re
import shutil
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import read, write
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from scoring.make_structures import mol_with_atom_index, remove_N2, remove_NH3

from .auto import shell
from .xyz2mol import read_xyz_file, xyz2AC, xyz2mol

file = "templates/core_noHS.mol"
core = Chem.MolFromMolFile(file, removeHs=False, sanitize=False)
"""Mol: 
mol object of the Mo core with dummy atoms instead of ligands
"""


def run_xtb(args):
    """Submit xtb calculations with given params

    Args:
        args (tuple): runner parameters

    Returns:
        results: Consists of energy and geometry of calculated structure
    """
    xyz_file, xtb_cmd, numThreads, conf_path, logname = args
    print(f"running {xyz_file} on {numThreads} core(s) starting at {datetime.now()}")

    cwd = os.path.dirname(xyz_file)
    xyz_file = os.path.basename(xyz_file)
    cmd = f"{xtb_cmd} -- {xyz_file} "
    os.environ["OMP_NUM_THREADS"] = f"{numThreads}"
    os.environ["MKL_NUM_THREADS"] = f"{numThreads}"
    os.environ["OMP_STACKSIZE"] = "1G"

    popen = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=False,
        cwd=cwd,
    )

    # Hardcoded wait time. Prevent an unfinished conformer from ruining the whole batch.
    try:
        output, err = popen.communicate(timeout=8 * 60)
    except subprocess.TimeoutExpired:
        popen.kill()
        output, err = popen.communicate()
    # Save logfiles
    with open(Path(conf_path) / f"{logname}job.out", "w") as f:
        f.write(output)
    with open(Path(conf_path) / f"{logname}err.out", "w") as f:
        f.write(err)

    results = read_results(output, err)

    return results


def write_to_db(database_dir=None, logfiles=None, trajfile=None):

    with connect(database_dir) as db:
        for i, (traj, logfile) in enumerate(zip(trajfile, logfiles)):

            energies = extract_energyxtb(logfile)
            structs = read(traj, index=":")
            for struct, energy in zip(structs, energies):
                id = db.reserve(name=str(traj) + str(i))
                if id is None:
                    continue
                struct.calc = SinglePointCalculator(struct, energy=energy)
                db.write(struct, id=id, name=str(traj))

    return


def extract_energyxtb(logfile=None):
    """
    Extracts xtb energies from xtb logfile using regex matching.

    Args:
        logfile (str): Specifies logfile to pull energy from

    Returns:
        energy (List[float]): List of floats containing the energy in each step
    """

    re_energy = re.compile("energy: (-\\d+\\.\\d+)")
    energy = []
    with logfile.open() as f:
        for line in f:
            if "energy" in line:
                energy.append(float(re_energy.search(line).groups()[0]))
    return energy


def create_intermediates(file=None, charge=0):
    """Create cycle where X HIPT groups are removed"""

    # Load xyz file and turn to mole object
    # Currently this does not work. Valence complains when i use the xyz file
    # atoms, _, coordinates = read_xyz_file(file)
    # new_mol = xyz2mol(atoms, coordinates, charge)

    mol = Chem.MolFromMolFile(
        file,
        sanitize=False,
        removeHs=False,
    )

    # Smart for a nitrogen bound to aromatic carbon.
    smart = "[c][N]"

    # Initialize pattern
    patt = Chem.MolFromSmarts(smart)

    # Substructure match
    print(f"Has substructure match: {mol.HasSubstructMatch(patt)}")
    indices = mol.GetSubstructMatches(patt)

    bonds = []
    # Cut the bonds between the nitrogen and the carbon.

    for tuple in indices:
        bonds.append(mol.GetBondBetweenAtoms(tuple[0], tuple[1]).GetIdx())

    # Cut
    frag = Chem.FragmentOnBonds(
        mol, bonds, addDummies=True, dummyLabels=[(1, 1), (1, 1), (1, 1)]
    )

    # Split mol object into individual fragments. sanitizeFrags needs to be false,
    # otherwise not work.
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # Substructure match to find the fragment with the Mo-core
    smart = "[Mo]"

    # Initialize pattern
    patt = Chem.MolFromSmarts(smart)

    # Substructure match
    for idx, struct in enumerate(frags):
        if struct.HasSubstructMatch(patt):
            print(f"Found the molybdenum core at index: {idx}")
            break

    # Replace dummy with hydrogen in the frag:
    for a in frags[idx].GetAtoms():
        if a.GetSymbol() == "*":
            a.SetAtomicNum(1)

    # Save frag to file
    fragname = "no_HIPT_frag.mol"
    with open(fragname, "w+") as f:
        f.write(Chem.MolToMolBlock(frags[idx]))
    return fragname


def read_results(output, err):
    if not "normal termination" in err:
        return 9999, {"atoms": None, "coords": None}
    lines = output.splitlines()
    energy = None
    structure_block = False
    atoms = []
    coords = []
    for l in lines:
        if "final structure" in l:
            structure_block = True
        elif structure_block:
            s = l.split()
            if len(s) == 4:
                atoms.append(s[0])
                coords.append(list(map(float, s[1:])))
            elif len(s) == 0:
                structure_block = False
        elif "TOTAL ENERGY" in l:
            energy = float(l.split()[3])
    return energy, {"atoms": atoms, "coords": coords}


def check_bonds(bare, charge, conf_paths, geometries, minidx, mol, xyzcoordinates):
    """Check for broken/formed bonds in the optimization

    Args:
        bare (bool): Make file with bare Mo and ligand
        charge (charge): Charge of molecule for xyz2mol
        conf_paths (Path): Path to optimized conformers
        geometries List(dict): Optimized geometries
        minidx (int): Idx of miniumum energy conformer
        mol (Chem.rdchem.Mol): Starting mol object
        xyzcoordinates (bool): Read xyz coordinates from xtb file

    Returns:
        final_geom : Geometry of min energy conformer
        bond_change (bool): Indicates whether a bond was broken/formed
    """

    # Create Mo and ligand bare xyz file
    if bare:
        file_bare = conf_paths[minidx] + f"/xtbopt_bare.xyz"
        tmp_mol = remove_NH3(mol)
        tmp_mol = remove_N2(tmp_mol)
        Chem.AddHs(tmp_mol)

        with open(file_bare, "w") as f:
            try:
                f.write(Chem.MolToXYZBlock(tmp_mol, confId=minidx.item()))
            except ValueError:
                print("Something happened with the conformer id")

    # Initialize paths
    file = conf_paths[minidx] + f"/xtbopt.xyz"
    file_noMo = conf_paths[minidx] + "/xtbopt_noMo.xyz"

    # Alter xyz file to remove the Mo for xyz2mol
    with open(file, "r") as file_input:
        with open(file_noMo, "w") as output:
            lines = file_input.readlines()
            new_str = str(int(lines[0]) - 1) + "\n"
            lines[0] = new_str
            for i, line in enumerate(lines):
                if "Mo " in line:
                    lines.pop(i)
                    pass
            output.writelines(lines)
    atoms, _, coordinates = read_xyz_file(file_noMo)
    print("Getting the adjacency matrices")
    AC, opt_mol = xyz2AC(atoms, coordinates, charge, use_huckel=True)

    # Check pre and after adjacency matrix
    before_ac = rdmolops.GetAdjacencyMatrix(mol)

    # Remove the Mo row:
    idx = mol.GetSubstructMatch(Chem.MolFromSmarts("[Mo]"))[0]
    intermediate = np.delete(before_ac, idx, axis=0)
    before_ac = np.delete(intermediate, idx, axis=1)

    # Check if any atoms have changed bonds
    bond_change = False
    if not np.all(before_ac == AC):
        print(
            f"There have been bonds changes. Saving struct and setting energy to 9999, for ligand {conf_paths[0]}"
        )
        bond_change = True

    # Get xyz coordnates or tuple
    if xyzcoordinates:
        with open(file, "r") as f:
            final_geom = f.readlines()
    else:
        final_geom = geometries[minidx]
    # Create traj file ready to write to database
    logfile = Path(conf_paths[minidx] + f"/xtbopt.log")
    trajfile = Path(conf_paths[minidx] + f"/traj.xyz")
    shutil.copy(logfile, trajfile)

    return final_geom, bond_change


# TODO change to new class format
def xtb_optimize_schrock(
    args,
    gbsa="benzene",
    alpb=None,
    opt_level="tight",
    input=None,
    name=None,
    cleanup=False,
    method=" 2",
):
    """Depcrecated optimization function, should be changed to class format"""

    files, parameters, numThreads, run_dir = args
    if not name:
        name = "tmp_" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=4)
        )
    # set SCRATCH if environmental variable
    try:
        scr_dir = os.environ["SCRATCH"]
    except:
        scr_dir = os.getcwd()
    print(f"SCRATCH DIR = {scr_dir}")

    # Get number of structs to optimize for parallellization
    n_structs = len(files)
    workers = np.min([numThreads, n_structs])

    # Perform initial ff optimization
    xtb_string = "xtb --gfnff --opt"
    args = [
        (str(xyz_file), xtb_string, 1, xyz_file.parent, "xtb_ff")
        for i, xyz_file in enumerate(files)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(run_xtb, args)

    # Prepare constrained opt string
    cmd = []
    xtb_string = f"xtb --gfn{method}"
    for elem in files:
        intermediate_name = elem.parent.name
        # Get intermediate parameters from the dict
        charge = parameters[intermediate_name]["charge"]
        spin = parameters[intermediate_name]["spin"]

        # xtb options
        XTB_OPTIONS = {
            "opt": opt_level,
            "chrg": charge,
            "uhf": spin,
            "gbsa": gbsa,
            "input": input,
        }

        for key, value in XTB_OPTIONS.items():
            xtb_string += f" --{key} {value}"
        cmd.append(xtb_string)
        xtb_string = "xtb"

    print("lol")

    xtbopt_files = [Path(xyz_file).parent / "xtbopt.xyz" for xyz_file in files]
    args = [
        (str(xyz_file), cmd[i], 1, xyz_file.parent, "xtb_con")
        for i, xyz_file in enumerate(xtbopt_files)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results2 = executor.map(run_xtb, args)

    # Get energies and geometries from final opt
    energies = []
    geometries = []
    for e, g in results2:
        energies.append(e)
        geometries.append(g)

    # Clean up
    if cleanup:
        shutil.rmtree(name)

    return energies, geometries


if __name__ == "__main__":
    # Debugging
    struct = ".xyz"
    paths = get_paths_molsimplify(source=args.cycle_dir, struct=struct, dest=dest)

    xtb_optimize_schrock(
        files=paths,
        parameters=parameters,
        gbsa="benzene",
        alpb=None,
        opt_level="tight",
        input=None,
        name=None,
        cleanup=False,
        numThreads=1,
    )