import argparse
import copy
import logging
import os
import pathlib
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
import pandas as pd
from rdkit import Chem
from ast import literal_eval as make_tuple

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))

from my_utils.classes import Conformers, Individual
from scoring import scoring_functions as sc
from scoring.scoring import (
    rdkit_embed_scoring,
    rdkit_embed_scoring_NH3plustoNH3,
    rdkit_embed_scoring_NH3toN2,
)

def get_arguments(arg_list=None):
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided.
        Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the arguments

    """
    parser = argparse.ArgumentParser(
        description="Run conformer screeening", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random initialization of population",
    )
    parser.add_argument(
        "--n_confs",
        type=int,
        default=1,
        help="How many conformers to generate",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=2,
        help="Number of cores to distribute over",
    )
    parser.add_argument(
        "--RMS_thresh",
        type=float,
        default=0.25,
        help="RMS pruning in embedding",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="RMS pruning in embedding",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=".",
        help="Directory to put various files",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="2",
        help="gfn method to use",
    )
    parser.add_argument("--bond_opt", action="store_true")
    # XTB specific params
    parser.add_argument(
        "--opt",
        type=str,
        default="tight",
        help="Opt convergence criteria for XTB",
    )
    parser.add_argument(
        "--gbsa",
        type=str,
        default="benzene",
        help="Type of solvent",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./xcontrol.inp",
        help="Name of input file that is created",
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
        "--file",
        type=Path,
        default="data/first_conformers.csv",
        help="File to get mol objects from"
    )
    return parser.parse_args(arg_list)



def get_start_population_from_csv(file=None, scoring=None):

    # Get ligands found with specified scoring function
    df = pd.read_csv(file, index_col=[0])
    df = df[df['scoring']==scoring]

    # Get list of mol objects
    inds = [Individual(Chem.MolFromSmiles(x), cut_idx=int(cut_idx), idx = make_tuple(idx)) for x, cut_idx, idx in zip(df['smiles'], df['cut_idx'], df.index)]

    # Initialize population object.
    conformers = Conformers(inds)

    return conformers


def main():

    # Get args
    args = get_arguments()

    # Create output folder
    args.output_dir.mkdir(exist_ok=True)

    # a dictionary mapping strings of function names to function objects:
    funcs = {
        "rdkit_embed_scoring": rdkit_embed_scoring,
        "rdkit_embed_scoring_NH3toN2": rdkit_embed_scoring_NH3toN2,
        "rdkit_embed_scoring_NH3plustoNH3": rdkit_embed_scoring_NH3plustoNH3,
    }

    # Get arguments as dict and add scoring function to dict.
    args_dict = vars(args)
    scoring_function = funcs[args.scoring_function]

    # Get start population from csv file
    conformers = get_start_population_from_csv(file=args.file, scoring=args.scoring_function)

    # Submit population for scoring with many conformers
    results = sc.slurm_scoring(scoring_function, conformers, args_dict)
    conformers.set_results(results)

    # Save the results:
    conformers.save(directory=args.output_dir, name=f"Conformers.pkl")

    print('Done')

if __name__ == '__main__':
    main()