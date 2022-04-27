"""

This is the driver script for doing various molS operations on the Mo catalyst core

Example:
    How to run:

        $ python molSymplify_driver.py args

"""

import pathlib
import sys, os
from pathlib import Path
import json, shutil, argparse, glob, time, pickle
from multiprocessing import Pool
from contextlib import suppress

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rdkit import Chem
from rdkit.Chem import Draw
from my_utils.my_utils import cd
from my_utils.my_xtb_utils import run_xtb, xtb_optimize_schrock
from my_utils.auto import shell, get_paths_custom, get_paths_molsimplify
from scoring.make_structures import create_dummy_ligand, connect_ligand, embed_rdkit
from scoring import scoring_functions as sc

import concurrent.futures


file = "templates/core_dummy.sdf"
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
        default="molS_drivers/Runs",
        help="Sets the output dir of the molSimplify commands",
    )
    parser.add_argument(
        "--cycle_dir",
        type=pathlib.Path,
        default="molS_drivers/Runs/Runs_cycle",
        help="Sets the output dir of the cycle molSimplify commands",
    )
    parser.add_argument(
        "--gen_path",
        type=pathlib.Path,
        default="/groups/kemi/magstr/GB_GA/generation_testbare/GA01.pkl",
        help="A generation pickle to load candidates from",
    )
    parser.add_argument(
        "--bare_struct",
        type=pathlib.Path,
        default="/home/magstr/Documents/GB_GA/debug/069_040_Mo_N2_NH3/conf001/xtbopt_bare.xyz",
        help="The structure with ligand and bare Mo",
    )
    parser.add_argument(
        "--cleanup",
        dest="cleanup",
        action="store_true",
        help="Cleans xtb output folders",
    )
    parser.add_argument('--no_cleanup', dest='cleanup', action='store_false')
    parser.set_defaults(cleanup=True)
    parser.add_argument(
        "--create_cycle",
        dest="create_cycle",
        action="store_true",
        help="Variable to control if cycle is created with ligand",
    )
    parser.add_argument(
        "--no_create_cycle",
        dest="create_cycle",
        action="store_false",
        help="Dont create cycle with molS",
    )
    parser.set_defaults(create_cycle=True)
    parser.add_argument(
        "--ligand_smi",
        type=str,
        default="NS(=O)(=O)N1CC1(Br)C#CS(=O)(=O)NBr",
        help="Set the ligand to put on the core",
    )
    parser.add_argument(
        "--xtbout",
        type=pathlib.Path,
        default="molS_drivers/xtbout",
        help="Directory to put xtb calculations",
    )
    parser.add_argument(
        "--log_tmp",
        type=pathlib.Path,
        default="molS_drivers/tmp_log",
        help="Directory to logfiles",
    )
    parser.add_argument(
        "--cut_idx",
        type=int,
        default=9,
        help="What primary amine to cut on",
    )
    parser.add_argument(
        "--starting_struct",
        type=str,
        default="GA_bare",
        help="Where to get the starting struct from",
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
        default=8,
        help="How many cores to use for the xtb",
    )
    return parser.parse_args(arg_list)


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    return mol


def create_cycleMS(
    new_core=None, smi_path=None, run_dir=None, log_tmp=None, ncores=None
):
    """
    Creates all the intermediates for the cycle based on the proposed catalyst.

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

    # Create logfolder
    (run_dir / log_tmp).mkdir(parents=True, exist_ok=True)

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
        results = executor.map(shell, args)

    # Check for badjobs
    if sorted(run_dir.rglob("*badjob*")):
        print("WARNING: Some jobs didnt not complete correctly with MolS")

    # Finaly create directory that contains the simple core
    # (for consistentxtb calcs)
    bare_core_dir = run_dir / "intermediate_Mo" / "struct"
    bare_core_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(new_core, bare_core_dir)

    return


def xtb_calc_serial(cycle_dir=None, param_path=None, dest="xtbout"):
    """

    Args:
        cycle_dir (str): The directory where the shcrock cycle was created with molS
        param_path (str): The path to the intermediate parameter dict.
        dest (str): Path to the output folder for the xtb calcs.

    Returns:
        None
    """

    # Search the cycle dir for xyz files and create xtb output folder.
    struct = ".xyz"
    paths = get_paths_molsimplify(source=cycle_dir, struct=struct, dest=dest)

    # Get spin and charge dict
    with open(param_path) as f:
        parameters = json.load(f)

    print(f"Optimizing at following locations: {paths}")
    for elem in paths:
        print(f"Processing {elem}")
        with cd(elem.parent):

            # Get intermediate name based on folder for path
            intermediate_name = elem.parent.name

            # Get intermediate parameters from the dict
            charge = parameters[intermediate_name]["charge"]
            spin = parameters[intermediate_name]["spin"]

            # Run the xtb calculation on the cut molecule
            run_xtb_my(
                structure=elem.name,
                method="gfn2",
                type="opt",
                charge=charge,
                spin=spin,
                gbsa="Benzene",
                numThreads=1,
            )
            i = 1
            if i == 1:
                print("lol")
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
    connect_idx = [x.GetIdx() for x in atom.GetNeighbors()]

    # Finaly replace the dummy with a hydrogen and print back to smiles
    for a in mol.GetAtoms():
        if a.GetSymbol() == "*":
            a.SetAtomicNum(1)

    new_lig_smi = Chem.MolToSmiles(mol)

    # We add one to the idx as molS is 1-indexed
    return connect_idx[0], new_lig_smi


def create_custom_core_rdkit(ligand, newcore_path='new_core.xyz'):


    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    catalyst = connect_ligand(core[0], ligand_cut)

    # Embed catalyst
    catalyst_3d = embed_rdkit(
        mol=catalyst,
        core=core[0],
        numConfs=1,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    # Save mol
    with open(newcore_path, "w+") as f:
        f.write(Chem.MolToXYZBlock(catalyst_3d))

    return newcore_path


def extract_structure_scoring(ligand):
    """
    Get the structure and prepare
    """
    structure_txt = ligand.structure
    newcore_path = "new_core.xyz"
    with open(newcore_path, "w+") as f:
        for line in structure_txt:
            f.write(line)

    return newcore_path


def create_custom_core(args):
    """

    Create a custom core based on a proposed ligand.

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


def main():
    """
    Driver script

    Returns:
        None
    """

    args = get_arguments()

    # Remove old cycle if dir exists, to prevent molS error.

    dirpath = args.run_dir
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    # Get the starting core. Either form rdkit or from optimization
    if args.starting_struct == "GA_bare":
        # extract_structure(ligand)
        shutil.copy(args.bare_struct, args.newcore_path)
    else:
        # Extract ligand object from GA.
        with open(args.gen_path, "rb") as f:
            gen = pickle.load(f)
        # TODO add option to select ligand by idx, currently selecting best one
        ligand = gen.survivors.molecules[-1]
        newcore_path = create_custom_core_rdkit(ligand, args)

    # Get path object for parameter dict
    intermediate_smi_path = Path("data/intermediate_smiles.json")

    if args.create_cycle:
        # Pass the output structure to cycle creation
        scoring_args = {
            "new_core": newcore_path,
            "smi_path": intermediate_smi_path,
            "run_dir": args.cycle_dir,
            "log_tmp": args.log_tmp,
            "ncores": args.ncores,
        }

        results = sc.slurm_scoring_molS(create_cycleMS, scoring_args)

    # Create xtb outpout folder
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dest = args.xtbout / (args.ligand_smi + timestr)
    dest.mkdir(parents=True, exist_ok=False)

    # Get spin and charge dict
    with open(intermediate_smi_path) as f:
        parameters = json.load(f)

    # Create new dir for xtb calcs and get paths
    struct = ".xyz"
    paths = get_paths_molsimplify(source=args.cycle_dir, struct=struct, dest=dest)

    # Start xtb calcs
    xtb_args = [paths, parameters, args.ncores, args.run_dir]
    results = sc.slurm_scoring_molS_xtb(xtb_optimize_schrock, xtb_args)

    if args.cleanup:
        collect_logfiles(dest=dest)

    sys.exit(0)


if __name__ == "__main__":
    # intermediate_smi_path = Path("../data/intermediate_smiles.json")
    # cycle_dir = "Runs/Runs_cycle"
    # create_cycleMS(
    #    new_core="../templates/core_withHS.xyz",
    #    smi_path=intermediate_smi_path,
    #    run_dir=cycle_dir,
    # )
    main()
