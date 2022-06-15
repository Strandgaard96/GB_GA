import re
import subprocess
import sys, os
import numpy as np
from pathlib import Path
import shutil
import argparse
import pickle
import json

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from my_utils.auto import shell_pure
from my_utils.my_utils import cd

# Dict for mapping options to input string
ORCA_COMMANDS = {
    "sp": "!PBE D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J def2/J MiniPrint KDIIS SOSCF",
    "opt": "!PBE D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J def2/J MiniPrint KDIIS SOSCF OPT",
    "freq": "!PBE D3BJ ZORA ZORA-def2-SVP SARC/J SPLIT-RI-J def2/J MiniPrint KDIIS SOSCF FREQ",
    "final_sp": "!B3LYP D3BJ ZORA ZORA-def2-TZVP SARC/J SPLIT-RI-J def2/J  RIJCOSX MiniPrint KDIIS SOSCF",
}

with open(
    os.path.join(source, "data/intermediate_smiles.json"), "r", encoding="utf-8"
) as f:
    smi_dict = json.load(f)


def get_inputfile(options, charge, spin, file):
    """Write Orca header"""

    header = "# ORCA input" + 2 * "\n"

    header += ORCA_COMMANDS[options.pop("type")] + "\n"
    header += (
        "%cpcm epsilon 1.844 end"
        + "\n"
        + '%basis\nNewGTO Mo "SARC-ZORA-TZVP" end\nend\n'
    )
    header += "# Number of cores\n"
    header += f"%pal nprocs {options.pop('n_cores')} end\n"
    header += "# RAM per core\n"
    header += f"%maxcore {1024 * options.pop('memory')}" + 2 * "\n"

    inputstr = header + 2 * "\n"

    # charge, spin, and coordinate section
    inputstr += f"* xyzfile {charge} {spin} {file}\n"
    inputstr += "\n"  # magic line

    return inputstr


def write_orca_input_file(
    orca_file="orca.inp",
    structure_path=None,
    command=ORCA_COMMANDS["sp"],
    charge=None,
    spin=None,
    n_cores=24,
):

    options = {"n_cores": n_cores, "memory": 8, "type": "sp"}

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
        default=".",
        help="Path to folder containing GA pickle files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=".",
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
        default=24,
        help="How many cores for each calc",
    )
    parser.add_argument(
        "--no_molecules",
        type=int,
        default=1,
        help="How many of the top molecules to do DFT on",
    )
    parser.add_argument("--function", choices=FUNCTION_MAP.keys())
    return parser.parse_args(arg_list)


def write_orca_sh(n_cores=24):

    # Copy template orca file from template dir
    shutil.copy(source / "dft/template_files/ORCA/orca.sh", ".")

    # Set node mem depending on n_core
    if n_cores == 24:
        mem = "250G"
    elif n_cores == 40:
        mem = "350G"

    with open("orca.sh", "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open("orca.sh", "w", encoding="utf-8") as f:
        for line in lines:
            if "--partition" in line:
                f.writelines(f"#SBATCH --partition=xeon{n_cores}\n")
            elif "#SBATCH -n" in line:
                f.writelines(f"#SBATCH -n {n_cores}\n")
            elif "#SBATCH --mem" in line:
                f.writelines(f"#SBATCH --mem={mem}\n")
            else:
                f.writelines(line)


def GA_singlepoints(args):
    """Future function that do singlpoints on GA objects"""

    # Extract dirs
    GA_dir = args.GA_dir
    output_dir = args.output_dir

    # Get how many max generations there are
    generation_no = len(sorted(GA_dir.glob("GA*.pkl"))) - 1

    # Possibly add option to select specific generation

    # Load GA object
    with open(GA_dir / f"GA{generation_no:0>2}.pkl", "rb") as f:
        gen = pickle.load(f)

    # Loop over all the structures
    for elem in gen.survivors.molecules[0 : args.no_molecules]:

        # TODO THE ORDERING OF THE KEYS MATTER HERE
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
        mol_dir1 = output_dir / f"{idx}_{key1}"
        mol_dir1.mkdir(exist_ok=True)

        # Create folders based on idx and intermediates
        mol_dir2 = output_dir / f"{idx}_{key2}"
        mol_dir2.mkdir(exist_ok=True)

        # xyzfile ame
        xyzfile = "struct.xyz"
        with cd(mol_dir1):
            # Create xtb input file from struct
            with open(xyzfile, "w+") as f:
                if elem.structure:
                    f.writelines(elem.structure)
                else:
                    print(f"No structure exists for this molecule: {elem.idx}")

            # Create input file
            write_orca_input_file(
                structure_path=xyzfile,
                command=ORCA_COMMANDS["sp"],
                charge=smi_dict[key1]["charge"],
                spin=smi_dict[key1]["mul"],
                n_cores=args.n_cores,
            )

            # Customize orca.sh to current job.
            write_orca_sh(n_cores=args.n_cores)

            cmd = "sbatch orca.sh"
            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)

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
                command=ORCA_COMMANDS["sp"],
                charge=smi_dict[key2]["charge"],
                spin=smi_dict[key2]["mul"],
                n_cores=args.n_cores,
            )

            # Customize orca.sh to current job.
            write_orca_sh(n_cores=args.n_cores)

            cmd = "sbatch orca.sh"
            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)

    return


def folder_orca_driver(args):
    """Future function that do DFT on structures in a folder"""

    # Extract dirs
    calc_dir = args.calc_dir
    output_dir = args.output_dir

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
                command=ORCA_COMMANDS["sp"],
                charge=smi_dict[key]["charge"],
                spin=smi_dict[key]["mul"],
                n_cores=args.n_cores,
            )

            # Customize orca.sh to current job.
            write_orca_sh(n_cores=args.n_cores)

            cmd = "sbatch orca.sh"
            # Submit bash script in folder
            out, err = shell_pure(cmd, shell=True)


if __name__ == "__main__":

    FUNCTION_MAP = {
        "folder_orca_driver": folder_orca_driver,
        "GA_singlepoints": GA_singlepoints,
    }
    args = get_arguments()
    func = FUNCTION_MAP[args.function]

    # Create output folder
    args.output_dir.mkdir(exist_ok=True)

    # Run chosen function
    func(args)

    sys.exit(0)
