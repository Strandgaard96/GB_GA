"""This is the driver script for doing various molS operations on the Mo
catalyst core.

Example:
    How to run:

        $ python molSymplify_driver.py args
"""

import argparse
import json
import os
import pathlib
import shutil
import time
from contextlib import suppress
from pathlib import Path

from rdkit import Chem
from support_mvp.auto import shell

from my_utils.data_utils import renamed_load
from my_utils.xtb_utils import xyz_to_conformer
from scoring.make_structures import (
    addAtom,
    connect_ligand,
    create_dummy_ligand,
    embed_rdkit,
    remove_N2,
    remove_NH3,
)

dirname = os.path.dirname(__file__)

source = Path(os.path.abspath(os.path.join(dirname, "../data")))
file = str(source / "templates/core_dummy.sdf")
core = Chem.SDMolSupplier(file, removeHs=False, sanitize=False)
"""Mol:
mol object of the Mo core with dummy atoms instead of ligands
"""


def get_arguments(arg_list=None):
    """
    Args:
        arg_list: Automatically obtained from the commandline if provided. Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the arguments

    """
    parser = argparse.ArgumentParser(
        description="Run GA algorithm", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--run_dir",
        type=pathlib.Path,
        default="Runs",
        help="Sets the output dir of the molSimplify commands",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default="cycle_creation",
        help="Sets the output dir of the cycle creation using RdKit",
    )
    parser.add_argument(
        "--cycle_dir",
        type=pathlib.Path,
        default="Runs/Runs_cycle",
        help="Sets the output dir of the cycle molSimplify commands",
    )
    parser.add_argument(
        "--pickle_path",
        type=pathlib.Path,
        default="../data/final_dft_opt.pkl",
        help="A pickle to load candidates from",
    )
    parser.add_argument(
        "--initial_opt",
        type=pathlib.Path,
        default="initial_opt/newcore.xyz",
        help="Where to perform initial opt",
    )
    parser.add_argument(
        "--bare_struct",
        type=pathlib.Path,
        default="/home/magstr/Documents/GB_GA/debug/conf001/xtbopt_bare.xyz",
        help="The structure with ligand and bare Mo",
    )
    parser.add_argument("--cleanup", dest="cleanup", action="store_true")
    parser.add_argument(
        "--create_cycle",
        dest="create_cycle",
        action="store_true",
        help="Variable to control if cycle is created with ligand",
    )
    parser.add_argument(
        "--ligand_smi",
        type=str,
        default="test",
        help="Set the ligand to put on the core",
    )
    parser.add_argument(
        "--xtbout",
        type=pathlib.Path,
        default="molS_drivers/xtbout",
        help="Directory to put xtb calculations",
    )
    parser.add_argument(
        "--cut_idx",
        type=int,
        default=9,
        help="What primary amine to cut on",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Which molecule to select for cycle creation",
    )
    parser.add_argument(
        "--newcore_path",
        type=pathlib.Path,
        default="newcore.xyz",
        help="Where to get the starting struct from",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=3,
        help="How many cores to use for the xtb",
    )
    return parser.parse_args(arg_list)


def get_paths_molsimplify(source, struct, dest):
    """Get the paths for all molS results.

    Create new folder structure
    that contains suffix as folder name and
    Args:
        source (str): Folder to look for files with struct extention
        struct (str): files types to look for in source.
        dest (Path):
    Returns:
        paths List(Path): Returns path objects to xyz files in newly created tree.
    """

    paths = []

    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith(struct):

                # Get intermediate name. A bit ugly and could break.
                new_dir = root.split("/")[-2].split("intermediate_")[-1]
                # Crreate this folder
                os.mkdir(os.path.join(dest, new_dir))
                # Copy the xyz file into the new directory and append the new file path
                shutil.copy(
                    os.path.join(root, file),
                    os.path.join(dest, new_dir + "/struct.xyz"),
                )
                paths.append(Path(os.path.join(dest, new_dir + "/struct.xyz")))
    return paths


def create_cycleMS(new_core=None, smi_path=None, run_dir=None, ncores=None):
    """Creates all the intermediates for the cycle based on the proposed
    catalyst.

    Args:
        new_core (str): Path to the new core with the proposed ligand put on.
        smi_path (str): Path to the intermediate dict
        run_dir (str): Directory to put the molS calc in

    Returns:
        None
    """
    with open(smi_path, "r", encoding="utf-8") as f:
        smi_dict = json.load(f)
        smi_dict.pop("Mo")

    # Get the idx of the Mo atom
    with open(new_core, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Mo" in line:
                # cc atoms is 1 indexed. # Substract 2 for the two empty lines and add one to get 1 index
                ccatoms = i - 2 + 1
                break

    args = []
    for key, value in smi_dict.items():

        intermediate_cmd = (
            f"molsimplify -core {new_core} -lig {value['smi']} -ligocc 1"
            f" -ccatoms {ccatoms} -skipANN True -spin 1 -oxstate 3 -ffoption no"
            f" -smicat 1 -suff intermediate_{key} -rundir {run_dir}"
        )

        # Modify string if multiple ligands.
        # Which is the case for equatorial NN at Mo_N2_NH3.
        if key == "Mo_N2_NH3":
            intermediate_cmd = None
            intermediate_cmd = (
                f"molsimplify -core {new_core} -oxstate 3 -lig [NH3],N#N"
                f" -ligocc 1,1 -ligloc True -ccatoms {ccatoms},{ccatoms} -spin 1 -oxstate 3"
                f" -skipANN True -smicat [1],[1]"
                f" -ffoption no -suff intermediate_{key} -rundir {run_dir}"
            )
        args.append((intermediate_cmd, key))

    for elem in args:
        shell(elem)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
    #    results = executor.map(shell, args)

    # Check for badjobs
    badjob = sorted(run_dir.rglob("*badjob*"))
    if badjob:
        print("WARNING: Some jobs didnt not complete correctly with MolS")
        print(badjob)

    # Finaly create directory that contains the simple core
    # (for consistentxtb calcs)
    bare_core_dir = run_dir / "Mo" / "struct"
    bare_core_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(new_core, bare_core_dir / "struct.xyz")

    # Move logfiles into folder

    timestr = time.strftime("%Y%m%d-%H%M%S")

    log_path = Path("logfiles" + timestr)
    log_path.mkdir(exist_ok=True)
    logfiles = sorted(Path(".").glob("*job.out"))
    for file in logfiles:
        shutil.move(str(file), str(log_path))
    logfiles = sorted(Path(".").glob("*err.out"))
    for file in logfiles:
        shutil.move(str(file), str(log_path))

    return


def collect_logfiles(dest=None):
    """
    Args:
        dest:
    Returns:
    """
    log_path = dest / "logfiles"
    os.mkdir(log_path)

    logfiles = sorted(dest.rglob("*job.out"))
    for file in logfiles:
        folder = log_path / file.parents[0].name
        folder.mkdir(exist_ok=True)
        shutil.copy(str(file), str(folder))

    with suppress(FileNotFoundError):
        os.remove("new_core.xyz")
        os.remove("CLIinput.inp")

    return None


def get_smicat(lig_smi=None):
    """
    Get the index of the atom that should connect to the Mo core
    Args:
        lig_smi (str):
    Returns:
        idx (int)
    """
    mol = lig_smi
    dummy = Chem.MolFromSmiles("*")
    # Get the dummy atom and then its atom object
    dummy_idx = mol.GetSubstructMatch(Chem.MolFromSmiles("*"))
    atom = mol.GetAtomWithIdx(dummy_idx[0])

    # Get the neighbor the dummy atom has.
    connect_idx = [x.GetIdx() for x in atom.GetNeighbors()][0]

    # Finaly replace the dummy with a hydrogen and print back to smiles
    for a in mol.GetAtoms():
        if a.GetSymbol() == "*":
            a.SetAtomicNum(1)

    new_lig_smi = Chem.MolToSmiles(mol)

    # We add one to the idx as molS is 1-indexed
    return connect_idx, new_lig_smi


def create_custom_core_rdkit(ligand, args):

    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    catalyst = connect_ligand(core[0], ligand_cut)
    catalyst = Chem.AddHs(catalyst)

    # Embed catalyst
    catalyst_3d = embed_rdkit(
        mol=catalyst,
        core=core[0],
        numConfs=1,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    # Save mol
    with open(args.newcore_path, "w+") as f:
        f.write(Chem.MolToXYZBlock(catalyst_3d))
    return args.newcore_path


def extract_structure_scoring(ligand):
    """Get the optimized xtb xyz structure for the given ligand and print into
    xyz-file."""
    structure_txt = ligand.structure
    newcore_path = "new_core.xyz"
    with open(newcore_path, "w+") as f:
        for line in structure_txt:
            f.write(line)

    return newcore_path


def create_custom_core(args):
    """Create a custom core based on a proposed ligand.

    Args:
        args (Namespace): args from driver

    Returns:
    """
    lig_smi = args.ligand_smi
    core_file = Path("templates/core_withHS.xyz")
    # Flag for using replace feature
    replig = 1
    suffix = "core_custom"
    run_dir = args.run_dir
    core_cmd = (
        f"molsimplify -core {core_file} -lig {lig_smi} -replig {replig} -ligocc 3"
        f" -ccatoms 24, 25, 26 -skipANN True -spin 1 -oxstate 3"
        f" -coord 5 -keepHs no -smicat {smicat} -rundir {run_dir} -suff {suffix}"
    )
    print(f"String passed to shell: {core_cmd}")
    out, err = shell(
        core_cmd,
        shell=False,
    )
    with open(f"job_{lig_smi}.out", "w", encoding="utf-8") as f:
        f.write(out)
    with open(f"err_{lig_smi}.out", "w", encoding="utf-8") as f:
        f.write(err)

    return


def cycle_to_dft():

    args = get_arguments()

    # Create output dir
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Get path object for parameter dict
    intermediate_smi_path = source / Path("intermediate_smiles.json")
    # Get spin and charge dict
    with open(intermediate_smi_path) as f:
        parameters = json.load(f)

    # Extract ligand object from GA.
    with open(args.pickle_path, "rb") as f:
        obj = renamed_load(f)

    # Select which molecule to do:
    idx = args.idx
    ligand = obj.molecules[idx].rdkit_mol
    mol = obj.molecules[idx].optimized_mol1

    # Get the bare xyz file
    file_bare = f"mol_bare.xyz"

    # Remove previous conformers and add the 3D structure from dft op
    # tmp_mol.RemoveAllConformers()

    with open(file_bare, "w+") as f:
        for line in obj.molecules[idx].final_structs["Mo_NH3"]:
            f.write(line)

    mol.RemoveAllConformers()

    xyz_to_conformer(mol, file_bare)

    mol = remove_NH3(mol)
    mol = remove_N2(mol)
    mol = Chem.AddHs(mol)

    with open("core_base.xyz", "w+") as f:
        f.write(Chem.MolToXYZBlock(mol))

    # Create all the mols with the different intermediates.
    Mo_match = mol.GetSubstructMatches(Chem.MolFromSmarts("[Mo]"))[0][0]

    # Embed a structure with N2 to use as reference
    Mo_N, idx = addAtom(mol, Mo_match, atom_type="N", order="triple")
    Mo_N.UpdatePropertyCache()
    Mo_N = Chem.AddHs(Mo_N)

    Mo_N = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(Mo_N)))

    # Embed mol object
    Mo_N_3d = embed_rdkit(
        mol=Mo_N,
        core=mol,
    )

    # Save structres for the Mo_N states
    with open("Mo_N.xyz", "w+") as f:
        f.write(Chem.MolToXYZBlock(mol))

    tmp = Chem.RemoveHs(Mo_N_3d)

    # Create all the mols with the different intermediates.
    N_Mo_match = tmp.GetSubstructMatches(Chem.MolFromSmarts("[N;D1,!X3]"))[0][0]

    # Embed a structure with N2 to use as reference
    Mo_N2, idx = addAtom(tmp, N_Mo_match, atom_type="N", order="triple", flag=True)

    # Get the first N-Mo bond and change bond order
    Mo_N2.GetBondBetweenAtoms(18, 17).SetBondType(Chem.rdchem.BondType.SINGLE)

    # Cleanup before embedding
    Mo_N2.UpdatePropertyCache()
    Mo_N2 = Chem.AddHs(Mo_N2)
    Mo_N2_mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(Mo_N2)))

    # Embed mol object
    Mo_N_3d = embed_rdkit(
        mol=Mo_N2_mol,
        core=mol,
    )


def main():
    """Driver script.

    Returns:
        None
    """

    args = get_arguments()

    # Get path object for parameter dict
    intermediate_smi_path = source / Path("intermediate_smiles.json")
    # Get spin and charge dict
    with open(intermediate_smi_path) as f:
        parameters = json.load(f)

    # Extract ligand object from GA.
    with open(args.pickle_path, "rb") as f:
        obj = renamed_load(f)

    # idx for top ten mols
    idxs = [
        6,
        8,
        9,
        11,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        23,
        25,
        26,
        95,
        105,
        106,
        115,
        119,
        126,
        128,
        129,
        132,
        134,
        135,
        136,
        139,
        5,
        7,
        10,
        12,
        17,
        22,
        24,
        29,
        35,
        40,
        47,
        65,
        70,
        83,
        90,
    ]
    mol_list = [ind for i, ind in enumerate(obj.molecules) if i in idxs]
    for ligand in mol_list:
        print(ligand.smiles)

        # Remove old cycle if dir exists, to prevent molS error.
        dirpath = args.run_dir
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)

        # Create cycle dir
        Path(args.cycle_dir).mkdir(parents=True, exist_ok=True)

        # Print custom core to use and get path
        mol = ligand.optimized_mol1

        mol = remove_NH3(mol)
        mol = remove_N2(mol)
        mol = Chem.AddHs(mol)
        Mo_bare_dir = Path("core_base.xyz")
        with open("core_base.xyz", "w+") as f:
            f.write(Chem.MolToXYZBlock(mol))

        xyz_file = Mo_bare_dir.resolve()

        if args.create_cycle:
            # Pass the output structure to cycle creation
            scoring_args = {
                "new_core": Mo_bare_dir,
                "smi_path": intermediate_smi_path,
                "run_dir": args.cycle_dir,
                "ncores": args.ncores,
            }

            create_cycleMS(**scoring_args)

            # results = sc.slurm_molS(create_cycleMS, scoring_args)

        # Create xtb outpout folder
        timestr = time.strftime("%Y%m%d-%H%M%S")
        dest = Path("dft_folder_41_44") / (ligand.smiles)

        isExist = os.path.exists(dest)
        if not isExist:
            dest.mkdir(parents=True, exist_ok=False)

        # Create new dir for xtb calcs and get paths
        struct = ".xyz"
        paths = get_paths_molsimplify(source=args.cycle_dir, struct=struct, dest=dest)

        # Extremely quick and dirty creation of constrain files
        create_constrain_files(paths)


def create_constrain_files(paths=None):

    # Get path object for parameter dict
    intermediate_smi_path = source / Path("intermediate_smiles.json")
    # Get spin and charge dict
    with open(intermediate_smi_path) as f:
        parameters = json.load(f)

    for path in paths:
        tmp = Path(path)
        with open(tmp, "r") as f:
            data = f.readlines()

        num_added = parameters[tmp.parent.name]["num_added"]

        # Get constrain list
        num_atoms = int(data[0])
        diff = num_atoms - num_added + 1

        match = [idx for idx in range(1, diff) if idx != 8]

        with open(os.path.join(path.parent, "xcontrol.inp"), "w") as f:
            f.write("$fix\n")
            f.write(f' atoms: {",".join(map(str, match))}\n')
            f.write("$end\n")


if __name__ == "__main__":
    # intermediate_smi_path = Path("../data/intermediate_smiles.json")
    # cycle_dir = "Runs/Runs_cycle"
    # create_cycleMS(
    #    new_core="../templates/core_withHS.xyz",
    #    smi_path=intermediate_smi_path,
    #    run_dir=cycle_dir,
    # )
    main()
    # cycle_to_dft()
