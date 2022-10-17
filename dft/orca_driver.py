import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from my_utils.auto import shell_pure
from my_utils.utils import cd

# Dict for mapping options to input string
ORCA_COMMANDS = {
    "sp": "!PBE D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J MiniPrint KDIIS SOSCF",
    "sp_sarcJ": "!PBE D3BJ ZORA ZORA-def2-TZVP  SARC/J SPLIT-RI-J MiniPrint KDIIS SOSCF",
    "opt": "!PBE D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J MiniPrint KDIIS SOSCF OPT",
    "freq": "!PBE D3BJ ZORA ZORA-def2-SVP SARC/J SPLIT-RI-J NormalPrint KDIIS SOSCF FREQ",
    "final_sp": "!B3LYP D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J RIJCOSX MiniPrint KDIIS SOSCF",
}

# Get dict with intermediate variables
with open(
    os.path.join(source, "data/intermediate_smiles.json"), "r", encoding="utf-8"
) as f:
    smi_dict = json.load(f)


def get_inputfile(options, charge, spin, file):
    """Write Orca header"""

    header = "# ORCA input" + 2 * "\n"

    header += ORCA_COMMANDS[options.pop("type_calc")] + "\n"
    header += (
        "%cpcm epsilon 1.844 end"
        + "\n"
        + '%basis\nNewGTO Mo "SARC-ZORA-TZVP" end\nend\n'
    )
    n_cores = options.pop("n_cores")
    header += "# Number of cores\n"
    header += f"%pal nprocs {n_cores} end\n"
    header += "# RAM per core\n"

    # Memory per core
    memory = options.pop("memory")
    mem_per_core = memory / n_cores
    header += f"%maxcore {round(1024 * mem_per_core)}" + 2 * "\n"
    inputstr = header + 2 * "\n"

    # charge, spin, and coordinate section
    inputstr += f"* xyzfile {charge} {spin} {file}\n"
    inputstr += "\n"  # magic line

    return inputstr


def get_inputfile_simple(options, charge, spin, file):
    """Write Orca header"""

    header = "# ORCA input" + 2 * "\n"

    header += ORCA_COMMANDS[options.pop("type_calc")] + "\n"
    header += "%cpcm epsilon 1.844 end" + "\n"
    n_cores = options.pop("n_cores")
    header += "# Number of cores\n"
    header += f"%pal nprocs {n_cores} end\n"
    header += "# RAM per core\n"

    # Memory per core
    memory = options.pop("memory")
    mem_per_core = memory / n_cores
    header += f"%maxcore {round(1024 * mem_per_core)}" + 2 * "\n"
    inputstr = header + 2 * "\n"

    # charge, spin, and coordinate section
    inputstr += f"* xyzfile {charge} {spin} {file}\n"
    inputstr += "\n"  # magic line

    return inputstr


def write_orca_input_file(
    orca_file="orca.inp",
    structure_path=None,
    type_calc="sp",
    charge=None,
    spin=None,
    n_cores=24,
    memory=8,
):

    options = {"n_cores": n_cores, "memory": memory, "type_calc": type_calc}

    # write input file
    inputstr = get_inputfile(options, charge=charge, spin=spin, file=structure_path)

    with open(orca_file, "w") as f:
        f.write(inputstr)


def get_arguments(arg_list=None):
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided.
        Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the arguments

    """
    parser = argparse.ArgumentParser(
        description="Run Orca calculation", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--calc_dir",
        type=Path,
        default=".",
        help="Path to folder containing xyz files",
    )
    parser.add_argument(
        "--GA_dir",
        type=Path,
        default="/home/magstr/generation_data/prod_new9_0",
        help="Path to folder containing GA pickle files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="debug_dft",
        help="Path to folder to put the DFT output folders",
    )
    parser.add_argument(
        "--scoring_function",
        dest="scoring_function",
        choices=[
            "rdkit_embed_scoring",
            "rdkit_embed_scoring_NH3toN2",
            "rdkit_embed_scoring_NH3plustoNH3",
        ],
        required=True,
        help="""Choose one of the specified scoring functions to know the 
        intermediates in question""",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=40,
        help="How many cores for each calc",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=8,
        help="How many GB requested for each calc",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="xeon40",
        help="Which partition to run on",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="niflheim",
        help="Which cluster the calc is running on",
    )
    parser.add_argument(
        "--no_molecules",
        type=int,
        default=[0, 10],
        nargs="+",
        help="How many of the top molecules to do DFT on",
    )
    parser.add_argument("--function", choices=FUNCTION_MAP.keys())
    parser.add_argument(
        "--type_calc",
        dest="type_calc",
        choices=list(ORCA_COMMANDS.keys()),
        required=True,
        help="""Choose top line for input file""",
    )
    return parser.parse_args(arg_list)


def write_orca_sh(
    n_cores=24, mem="250G", partition="xeon40", name="orca", cluster="niflheim"
):

    # Copy template orca file from template dir
    if cluster == "niflheim":
        shutil.copy(source / "dft/template_files/ORCA/orca.sh", "./orca.sh")
    elif cluster == "steno":
        shutil.copy(source / "dft/template_files/ORCA/orca_steno.sh", "./orca.sh")
    else:
        print("Invalid cluster")

    with open("orca.sh", "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open("orca.sh", "w", encoding="utf-8") as f:
        for line in lines:
            if "--partition" in line:
                f.writelines(f"#SBATCH --partition={partition}\n")
            elif "#SBATCH -n" in line:
                f.writelines(f"#SBATCH -n {n_cores}\n")
            elif "#SBATCH --mem" in line:
                f.writelines(f"#SBATCH --mem={mem}GB\n")
            elif "#SBATCH --job_name" in line:
                f.writelines(f"#SBATCH --job-name={name}\n")
            elif "#SBATCH --time" in line:
                if "xeon16" in partition:
                    f.writelines(f"#SBATCH --time=7-00:00:00\n")
                else:
                    f.writelines("#SBATCH --time=2-02:00:00\n")
            else:
                f.writelines(line)


def GA_singlepoints(args):
    """Future function that do singlpoints on GA objects"""

    # Extract dirs
    GA_dir = args.GA_dir

    # Create output folder
    args.output_dir.mkdir(exist_ok=True)
    output_dir = args.output_dir

    # Get how many max generations there are
    generation_no = len(sorted(GA_dir.glob("GA[0-9][0-9].pkl"))) - 1

    # Possibly add option to select specific generation

    # Load GA object
    with open(GA_dir / f"GA{generation_no:0>2}.pkl", "rb") as f:
        gen = pickle.load(f)

    # Loop over all the structures
    for elem in gen.molecules[args.no_molecules[0] : args.no_molecules[1]]:

        # THE ORDERING OF THE KEYS MATTER HERE
        # Get scoring intermediates and charge/spin
        scoring = args.scoring_function
        if scoring == "rdkit_embed_scoring":
            key1 = "Mo_N2_NH3"
            key2 = "Mo_NH3"
        elif scoring == "rdkit_embed_scoring_NH3toN2":
            key1 = "Mo_N2"
            key2 = "Mo_NH3"
        elif scoring == "rdkit_embed_scoring_NH3plustoNH3":
            key1 = "Mo_NH3"
            key2 = "Mo_NH3+"

        # Format the idx
        idx = re.sub(r"[()]", "", str(elem.idx))
        idx = idx.replace(",", "_").replace(" ", "")

        # Create folders based on idx and intermediates
        mol_dir1 = output_dir / f"{idx}" / key1
        mol_dir1.mkdir(exist_ok=True, parents=True)

        # Create folders based on idx and intermediates
        mol_dir2 = output_dir / f"{idx}" / key2
        mol_dir2.mkdir(exist_ok=True, parents=True)

        # xyzfile ame
        xyzfile = "struct.xyz"
        with cd(mol_dir1):

            # Save indvidual object for easier processing later
            elem.save(directory=".")

            # Create xtb input file from struct
            with open(xyzfile, "w+") as f:
                if elem.structure:
                    f.writelines(elem.structure)
                else:
                    print(f"No structure exists for this molecule: {elem.idx}")

            # Create input file
            write_orca_input_file(
                structure_path=xyzfile,
                type_calc=args.type_calc,
                charge=smi_dict[key1]["charge"],
                spin=smi_dict[key1]["mul"],
                n_cores=args.n_cores,
                memory=args.memory,
            )

            # Customize orca.sh to current job.
            write_orca_sh(
                n_cores=args.n_cores,
                mem=args.memory,
                partition=args.partition,
                cluster=args.cluster,
            )

            cmd = "sbatch orca.sh"
            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)
            with open(f"1job.err", "w") as f:
                f.write(err)
            with open(f"1job.out", "w") as f:
                f.write(out)

        with cd(mol_dir2):
            # Create xtb input file from struct
            with open(xyzfile, "w+") as f:
                if elem.structure2:
                    f.writelines(elem.structure2)
                else:
                    print(f"No structure exists for this molecule: {elem.idx}")

            # Create input file
            write_orca_input_file(
                structure_path=xyzfile,
                type_calc=args.type_calc,
                charge=smi_dict[key2]["charge"],
                spin=smi_dict[key2]["mul"],
                n_cores=args.n_cores,
                memory=args.memory,
            )

            # Customize orca.sh to current job.
            write_orca_sh(
                n_cores=args.n_cores,
                mem=args.memory,
                partition=args.partition,
                cluster=args.cluster,
            )

            cmd = "sbatch orca.sh"
            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)
            with open(f"2job.err", "w") as f:
                f.write(err)
            with open(f"2job.out", "w") as f:
                f.write(out)

    return


def folder_orca_driver(args):
    """Future function that do DFT on structures in a folder"""

    # Extract dirs
    calc_dir = args.calc_dir

    # Get all structures
    paths = sorted(calc_dir.rglob("*.xyz"))

    # Loop over folders
    for path in paths:

        # Get the key for the current structure
        key = str(path.parent.name)

        with cd(path.parent):
            # Create input file
            write_orca_input_file(
                structure_path=path.name,
                type_calc=args.type_calc,
                charge=smi_dict[key]["charge"],
                spin=smi_dict[key]["mul"],
                n_cores=args.n_cores,
                memory=args.memory,
            )

            # Customize orca.sh to current job.
            write_orca_sh(
                n_cores=args.n_cores,
                mem=args.memory,
                partition=args.partition,
                cluster=args.cluster,
            )

            cmd = "sbatch orca.sh"

            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)
            with open(f"job.err", "w") as f:
                f.write(err)
            with open(f"job.out", "w") as f:
                f.write(out)


def conformersearch_dft_driver(args):

    """Do DFT optimization on structures based on conformer search"""

    # Directory for the conformer object
    calc_dir = args.calc_dir

    # Create dir for results
    output_dir_dft = args.output_dir / "dft"
    output_dir_dft.mkdir(exist_ok=True, parents=True)

    # Load GA object
    with open(calc_dir / f"Conformers.pkl", "rb") as f:
        conf = pickle.load(f)

    # Loop over all the structures with no bond changes
    for molecule in conf.molecules:

        # THE ORDERING OF THE KEYS MATTER HERE
        # Get scoring intermediates and charge/spin
        scoring = molecule.scoring_function
        if scoring == "rdkit_embed_scoring":
            key1 = "Mo_N2_NH3"
            key2 = "Mo_NH3"
        elif scoring == "rdkit_embed_scoring_NH3toN2":
            key2 = "Mo_NH3"
            key1 = "Mo_N2"
        elif scoring == "rdkit_embed_scoring_NH3plustoNH3":
            key2 = "Mo_NH3+"
            key1 = "Mo_NH3"

        # Format the idx
        idx = re.sub(r"[()]", "", str(molecule.idx))
        idx = idx.replace(",", "_").replace(" ", "")

        # Create folders based on idx and intermediates
        mol_dir1 = output_dir_dft / f"{idx}" / key1
        mol_dir1.mkdir(exist_ok=True, parents=True)

        # Create folders based on idx and intermediates
        mol_dir2 = output_dir_dft / f"{idx}" / key2
        mol_dir2.mkdir(exist_ok=True, parents=True)

        xyzfile = "struct.xyz"

        if not molecule.optimized_mol1:
            print(f"None for {molecule.idx}, {molecule.scoring_function}")
            continue

        # Resort conformers by energy
        confs = molecule.optimized_mol1.GetConformers()
        energies = molecule.energy_dict["energy1"]
        confs = [c for _, c in sorted(zip(energies, confs))]

        # Loop the conformer dirs
        for i, conf in enumerate(confs[args.no_molecules[0] : args.no_molecules[1]]):

            conf_dir1 = mol_dir1 / f"conf{i:03d}"
            conf_dir1.mkdir(exist_ok=True)

            with cd(conf_dir1):

                # Save indvidual object for easier processing later
                molecule.save(directory=".")

                # Create xtb input file from struct
                number_of_atoms = molecule.optimized_mol1.GetNumAtoms()
                symbols = [a.GetSymbol() for a in molecule.optimized_mol1.GetAtoms()]
                with open(xyzfile, "w") as _file:
                    _file.write(str(number_of_atoms) + "\n")
                    _file.write(
                        f"{Chem.MolToSmiles(Chem.RemoveHs(molecule.optimized_mol1))}\n"
                    )
                    for atom, symbol in enumerate(symbols):
                        p = conf.GetAtomPosition(atom)
                        line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                        _file.write(line)

                # Create input file
                write_orca_input_file(
                    structure_path=xyzfile,
                    type_calc=args.type_calc,
                    charge=smi_dict[key1]["charge"],
                    spin=smi_dict[key1]["mul"],
                    n_cores=args.n_cores,
                    memory=args.memory,
                )

                # Customize orca.sh to current job.
                write_orca_sh(
                    n_cores=args.n_cores,
                    mem=args.memory,
                    partition=args.partition,
                    cluster=args.cluster,
                )

                cmd = "sbatch orca.sh"

                # Submit bash script in folder
                out, err = shell_pure(cmd, shell=True)
                with open(f"job.err", "w") as f:
                    f.write(err)
                with open(f"job.out", "w") as f:
                    f.write(out)

        # Resort conformers by energy
        confs = molecule.optimized_mol2.GetConformers()
        energies = molecule.energy_dict["energy2"]
        confs = [c for _, c in sorted(zip(energies, confs))]

        for i, conf in enumerate(confs[args.no_molecules[0] : args.no_molecules[1]]):
            conf_dir2 = mol_dir2 / f"conf{i:03d}"
            conf_dir2.mkdir(exist_ok=True)

            with cd(conf_dir2):

                # Save indvidual object for easier processing later
                molecule.save(directory=".")

                number_of_atoms = molecule.optimized_mol2.GetNumAtoms()
                symbols = [a.GetSymbol() for a in molecule.optimized_mol2.GetAtoms()]
                # Create xtb input file from struct
                with open(xyzfile, "w") as _file:
                    _file.write(str(number_of_atoms) + "\n")
                    _file.write(f"{Chem.MolToSmiles(molecule.optimized_mol2)}\n")
                    for atom, symbol in enumerate(symbols):
                        p = conf.GetAtomPosition(atom)
                        line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                        _file.write(line)

                # Create input file
                write_orca_input_file(
                    structure_path=xyzfile,
                    type_calc=args.type_calc,
                    charge=smi_dict[key2]["charge"],
                    spin=smi_dict[key2]["mul"],
                    n_cores=args.n_cores,
                    memory=args.memory,
                )

                # Customize orca.sh to current job.
                write_orca_sh(
                    n_cores=args.n_cores,
                    mem=args.memory,
                    partition=args.partition,
                    cluster=args.cluster,
                )

                cmd = "sbatch orca.sh"

                # Submit bash script in folder
                out, err = shell_pure(cmd, shell=True)
                with open(f"job.err", "w") as f:
                    f.write(err)
                with open(f"job.out", "w") as f:
                    f.write(out)


def conformersearch_dft_driver_single_molecule(molecule, args):

    # THE ORDERING OF THE KEYS MATTER HERE
    # Get scoring intermediates and charge/spin
    scoring = molecule.scoring_function
    if scoring == "rdkit_embed_scoring":
        key1 = "Mo_N2_NH3"
        key2 = "Mo_NH3"
    elif scoring == "rdkit_embed_scoring_NH3toN2":
        key1 = "Mo_N2"
        key2 = "Mo_NH3"
    elif scoring == "rdkit_embed_scoring_NH3plustoNH3":
        key1 = "Mo_NH3"
        key2 = "Mo_NH3+"

    # Format the idx
    idx = re.sub(r"[()]", "", str(molecule.idx))
    idx = idx.replace(",", "_").replace(" ", "")

    # Create folders based on idx and intermediates
    mol_dir1 = output_dir_dft / f"{idx}" / key1
    mol_dir1.mkdir(exist_ok=True, parents=True)

    # Create folders based on idx and intermediates
    mol_dir2 = output_dir_dft / f"{idx}" / key2
    mol_dir2.mkdir(exist_ok=True, parents=True)

    xyzfile = "struct.xyz"

    if not molecule.optimized_mol1:
        print(f"None for {molecule.idx}, {molecule.scoring_function}")
        # continue

    # Resort conformers by energy
    confs = molecule.optimized_mol1.GetConformers()
    energies = molecule.energy_dict["energy1"]
    confs = [c for _, c in sorted(zip(energies, confs))]

    # Loop the conformer dirs
    for i, conf in enumerate(confs[args.no_molecules[0] : args.no_molecules[1]]):

        conf_dir1 = mol_dir1 / f"conf{i:03d}"
        conf_dir1.mkdir(exist_ok=True)

        with cd(conf_dir1):

            # Save indvidual object for easier processing later
            molecule.save(directory=".")

            # Create xtb input file from struct
            number_of_atoms = molecule.optimized_mol1.GetNumAtoms()
            symbols = [a.GetSymbol() for a in molecule.optimized_mol1.GetAtoms()]
            with open(xyzfile, "w") as _file:
                _file.write(str(number_of_atoms) + "\n")
                _file.write(
                    f"{Chem.MolToSmiles(Chem.RemoveHs(molecule.optimized_mol1))}\n"
                )
                for atom, symbol in enumerate(symbols):
                    p = conf.GetAtomPosition(atom)
                    line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                    _file.write(line)

            # Create input file
            write_orca_input_file(
                structure_path=xyzfile,
                type_calc=args.type_calc,
                charge=smi_dict[key1]["charge"],
                spin=smi_dict[key1]["mul"],
                n_cores=args.n_cores,
                memory=args.memory,
            )

            # Customize orca.sh to current job.
            write_orca_sh(
                n_cores=args.n_cores,
                mem=args.memory,
                partition=args.partition,
                cluster=args.cluster,
            )

            cmd = "sbatch orca.sh"

            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)
            with open(f"job.err", "w") as f:
                f.write(err)
            with open(f"job.out", "w") as f:
                f.write(out)

    # Resort conformers by energy
    confs = molecule.optimized_mol2.GetConformers()
    energies = molecule.energy_dict["energy2"]
    confs = [c for _, c in sorted(zip(energies, confs))]

    for i, conf in enumerate(confs[args.no_molecules[0] : args.no_molecules[1]]):
        conf_dir2 = mol_dir2 / f"conf{i:03d}"
        conf_dir2.mkdir(exist_ok=True)

        with cd(conf_dir2):

            # Save indvidual object for easier processing later
            molecule.save(directory=".")

            number_of_atoms = molecule.optimized_mol2.GetNumAtoms()
            symbols = [a.GetSymbol() for a in molecule.optimized_mol2.GetAtoms()]
            # Create xtb input file from struct
            with open(xyzfile, "w") as _file:
                _file.write(str(number_of_atoms) + "\n")
                _file.write(f"{Chem.MolToSmiles(molecule.optimized_mol2)}\n")
                for atom, symbol in enumerate(symbols):
                    p = conf.GetAtomPosition(atom)
                    line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                    _file.write(line)

            # Create input file
            write_orca_input_file(
                structure_path=xyzfile,
                type_calc=args.type_calc,
                charge=smi_dict[key2]["charge"],
                spin=smi_dict[key2]["mul"],
                n_cores=args.n_cores,
                memory=args.memory,
            )

            # Customize orca.sh to current job.
            write_orca_sh(
                n_cores=args.n_cores,
                mem=args.memory,
                partition=args.partition,
                cluster=args.cluster,
            )

            cmd = "sbatch orca.sh"

            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)
            with open(f"job.err", "w") as f:
                f.write(err)
            with open(f"job.out", "w") as f:
                f.write(out)


def parts_opts(args):
    """Future function that do DFT on structures in a folder"""

    # Extract dirs
    calc_dir = args.calc_dir

    # Get all structures
    paths = sorted(calc_dir.rglob("*.xyz"))

    # Loop over folders
    for path in paths:

        with cd(path.parent):
            # Get the charge ans spin from default.in file
            with open("default.in", "r") as f:
                data = f.readlines()

            charge = data[0].split("=")[-1].split("\n")[0]
            spin = data[1].split("=")[-1].split("\n")[0]
            # Create input file
            write_orca_input_file(
                structure_path=path.name,
                type_calc=args.type_calc,
                charge=charge,
                spin=spin,
                n_cores=args.n_cores,
                memory=args.memory,
            )

            # Customize orca.sh to current job.
            write_orca_sh(
                n_cores=args.n_cores,
                mem=args.memory,
                partition=args.partition,
                cluster=args.cluster,
            )

            cmd = "sbatch orca.sh"

            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)
            with open(f"job.err", "w") as f:
                f.write(err)
            with open(f"job.out", "w") as f:
                f.write(out)


if __name__ == "__main__":

    FUNCTION_MAP = {
        "folder_orca_driver": folder_orca_driver,
        "GA_singlepoints": GA_singlepoints,
        "parts_opts": parts_opts,
        "conformer_dft": conformersearch_dft_driver,
    }
    args = get_arguments()
    func = FUNCTION_MAP[args.function]

    # Run chosen function
    func(args)

    sys.exit(0)
