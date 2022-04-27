from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from .xyz2mol import read_xyz_file, xyz2mol, xyz2AC
from .auto import shell
from scoring.make_structures import remove_NH3, remove_N2, mol_with_atom_index

import os
import random
import numpy as np
import shutil
import string
import subprocess
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

import concurrent.futures


file = "templates/core_noHS.mol"
core = Chem.MolFromMolFile(file, removeHs=False, sanitize=False)
"""Mol: 
mol object of the Mo core with dummy atoms instead of ligands
"""


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    Chem.Draw.MolToImage(mol, size=(400, 400)).show()
    return mol


# %%
def write_xtb_input_files(fragment, name, destination="."):
    number_of_atoms = fragment.GetNumAtoms()
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
    conformers = fragment.GetConformers()
    file_paths = []
    conf_paths = []
    for i, conf in enumerate(conformers):
        conf_path = os.path.join(destination, f"conf{i:03d}")
        conf_paths.append(conf_path)

        if os.path.exists(conf_path):
            shutil.rmtree(conf_path)
        os.makedirs(conf_path)

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
    return file_paths, conf_paths


def run_xtb(args):
    xyz_file, xtb_cmd, numThreads, conf_path, logname = args
    print(f"running {xyz_file} on {numThreads} core(s) starting at {datetime.now()}")

    cwd = os.path.dirname(xyz_file)
    xyz_file = os.path.basename(xyz_file)
    cmd = f"{xtb_cmd} -- {xyz_file} "
    os.environ["OMP_NUM_THREADS"] = f"{numThreads}"
    os.environ["MKL_NUM_THREADS"] = f"{numThreads}"
    os.environ["OMP_STACKSIZE"] = "2G"

    popen = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=False,
        cwd=cwd,
    )

    output, err = popen.communicate()

    with open(Path(conf_path) / f"{logname}job.out", "w") as f:
        f.write(output)
    with open(Path(conf_path) / f"{logname}err.out", "w") as f:
        f.write(err)
    results = read_results(output, err)
    return results


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

    # Split mol object into individual fragments. sanitizeFrags needs to be false, otherwise not work.
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


def xtb_pre_optimize(
    mol,
    method="ff",
    charge=None,
    spin=None,
    gbsa="benzene",
    alpb=None,
    opt_level="tight",
    name=None,
    cleanup=False,
    preoptimize=True,
    numThreads=1,
    xyzcoordinates=True,
):
    # check mol input
    assert isinstance(mol, Chem.rdchem.Mol)
    if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
        raise Exception("Implicit Hydrogens")
    conformers = mol.GetConformers()
    n_confs = len(conformers)
    if not conformers:
        raise Exception("Mol is not embedded")
    elif not conformers[-1].Is3D():
        raise Exception("Conformer is not 3D")

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

    print("write input files")
    xyz_files, conf_paths = write_xtb_input_files(mol, "xtbmol", destination=name)

    # Make input constrain file
    if any("N2" in s for s in conf_paths):
        make_input_constrain_file(mol, core=core, path=conf_paths, NH3=True, N2=True)
    else:
        make_input_constrain_file(mol, core=core, path=conf_paths, NH3=True)

    # xtb options
    XTB_OPTIONS = {
        "opt": opt_level,
        "chrg": charge,
        "uhf": spin,
        "gbsa": gbsa,
        "input": "./xcontrol.inp",
    }

    cmd = f"xtb --gfn{method}"
    for key, value in XTB_OPTIONS.items():
        cmd += f" --{key} {value}"

    workers = numThreads
    cpus_per_worker = 1
    print(f"workers: {workers}, cpus_per_worker: {cpus_per_worker}")
    args = [
        (xyz_file, cmd, cpus_per_worker, conf_paths[i], "ff")
        for i, xyz_file in enumerate(xyz_files)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(run_xtb, args)

    # Store the log file
    for elem in conf_paths:
        shutil.copy(os.path.join(elem, "xtbopt.log"), os.path.join(elem, "ffopt.log"))

    if preoptimize:
        cmd = cmd.replace("gfnff", "gfn 2")
        xyz_files = [Path(xyz_file).parent / "xtbopt.xyz" for xyz_file in xyz_files]
        args = [
            (xyz_file, cmd, cpus_per_worker, conf_paths[i], "const_gfn2")
            for i, xyz_file in enumerate(xyz_files)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results2 = executor.map(run_xtb, args)

    # Store the log file
    for elem in conf_paths:
        shutil.copy(
            os.path.join(elem, "xtbopt.log"), os.path.join(elem, "constrained_opt.log")
        )

    # Perform final relaxation
    # cmd = cmd.replace("--input ./xcontrol.inp", "")

    # Constrain only N reactants and Mo
    if any("N2" in s for s in conf_paths):
        make_input_constrain_file(
            mol, core=Chem.MolFromSmiles("[Mo]"), path=conf_paths, NH3=True, N2=True
        )
    else:
        make_input_constrain_file(
            mol, core=Chem.MolFromSmiles("[Mo]"), path=conf_paths, NH3=True, N2=False
        )

    args = [
        (xyz_file, cmd, cpus_per_worker, conf_paths[i], "gfn2")
        for i, xyz_file in enumerate(xyz_files)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results3 = executor.map(run_xtb, args)

    print("Finished all optimizations")

    energies = []
    geometries = []
    for e, g in results3:
        energies.append(e)
        geometries.append(g)

    arr = np.array(energies, dtype=np.float)

    try:
        minidx = np.nanargmin(arr)
    except ValueError:
        print(
            "All-Nan slice encountered, setting minidx to None and returning 9999 energy"
        )
        energies = 9999
        geometries = None
        minidx = None
        return energies, geometries, minidx

    # Clean up
    if cleanup:
        shutil.rmtree(name)

    file_bare = conf_paths[minidx] + f"/xtbopt_bare.xyz"

    if "N2" in file_bare:
        tmp_mol = remove_NH3(mol)
        tmp_mol = remove_N2(tmp_mol)
        Chem.AddHs(tmp_mol)

        with open(file_bare, "w") as f:
            try:
                f.write(Chem.MolToXYZBlock(tmp_mol, confId=minidx.item()))
            except ValueError:
                print("Something happened with the conformer id")

    file = conf_paths[minidx] + f"/xtbopt.xyz"
    file_noMo = conf_paths[minidx] + "/xtbopt_noMo.xyz"

    # Alter xyz file
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

    print("Performing charge loop and xyz2mol")
    # Loop to check different charges. Very hardcoded and should maybe be changed

    print("Getting the adjacency matrices")
    AC, opt_mol = xyz2AC(atoms, coordinates, charge, use_huckel=True)

    # Check pre and after adjacency matrix
    before_ac = rdmolops.GetAdjacencyMatrix(mol)

    # Remove the Mo row:
    idx = mol.GetSubstructMatch(Chem.MolFromSmarts("[Mo]"))[0]
    intermediate = np.delete(before_ac, idx, axis=0)
    before_ac = np.delete(intermediate, idx, axis=1)

    # Check if any atoms have 0 bonds, then handle
    if not np.all(before_ac == AC):
        print(
            f"There have been bonds changes. Saving struct and setting energy to 9999, for ligand {conf_paths[0]}"
        )
        energies = 9999
        geometries = None
        minidx = None
        return energies, geometries, minidx

    # Get xyz coordnates or tuple
    if xyzcoordinates:
        with open(file, "r") as f:
            final_geom = f.readlines()
    else:
        final_geom = geometries[minidx]

    return energies[minidx], final_geom, minidx.item()


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


def make_input_constrain_file(molecule, core, path, NH3=False, N2=False):
    # Locate atoms to contrain
    match = (
        np.array(molecule.GetSubstructMatch(core)) + 1
    )  # indexing starts with 0 for RDKit but 1 for xTB
    match = sorted(match)
    assert len(match) == core.GetNumAtoms(), "ERROR! Complete match not found."

    if NH3:
        NH3_match = Chem.MolFromSmarts("[NH3]")
        NH3_match = Chem.AddHs(NH3_match)
        NH3_sub_match = np.array(molecule.GetSubstructMatch(NH3_match)) + 1
        match.extend(NH3_sub_match)
    if N2:
        N2_match = Chem.MolFromSmarts("N#N")
        N2_sub_match = np.array(molecule.GetSubstructMatch(N2_match)) + 1
        match.extend(N2_sub_match)

    for elem in path:
        # Write the xcontrol file
        with open(os.path.join(elem, "xcontrol.inp"), "w") as f:
            f.write("$fix\n")
            f.write(f' atoms: {",".join(map(str, match))}\n')
            f.write("$end\n")
    return


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
