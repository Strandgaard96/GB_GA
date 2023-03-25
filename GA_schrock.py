"""Driver script for running a GA algorithm.

Written by Magnus Strandgaard 2023

Example:
    How to run:

        $ python GA_schrock.py --args
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_logger = logging.getLogger(__name__)

import crossover as co
import filters
from GeneticAlgorithm import GeneticAlgorithm
from utils.utils import get_git_revision_short_hash

molecule_filter = filters.get_molecule_filters(None, "./filters/alert_collection.csv")


def get_arguments(arg_list=None):
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided.
        Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the driver arguments.

    """
    parser = argparse.ArgumentParser(
        description="Run GA algorithm", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=10,
        help="Sets the size of population pool.",
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
        default=2,
        help="How many conformers to generate",
    )
    parser.add_argument(
        "--n_tries",
        type=int,
        default=1,
        help="How many overall runs of the GA",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=2,
        help="Number of cores to distribute xTB over",
    )
    parser.add_argument(
        "--mem_per_cpu",
        type=str,
        default="500MB",
        help="Mem per cpu allocated for submitit job",
    )
    parser.add_argument(
        "--partition",
        choices=[
            "kemi1",
            "xeon24",
            "xeon40",
        ],
        required=True,
        help="""Choose partitoin to run on""",
    )
    parser.add_argument(
        "--RMS_thresh",
        type=float,
        default=0.25,
        help="RMS pruning in embedding",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=1,
        help="How many times is the population optimized",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.5,
        help="Mutation rate",
    )
    parser.add_argument(
        "--sa_screening",
        dest="sa_screening",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="data/ZINC_250k.smi",
        help="",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="debug/generation_debug",
        help="Directory to put output files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=12,
        help="Minutes before timeout in xTB optimization",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ga_scoring", action="store_true")
    parser.add_argument("--supress_amines", action="store_true")
    parser.add_argument(
        "--energy_cutoff",
        type=float,
        default=0.0159,
        help="Cutoff for conformer energies",
    )
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument(
        "--scoring_function",
        choices=["calculate_score_logP", "calculate_score"],
        required=True,
        help="""Choose one of the specified scoring functions to be run.""",
    )
    # XTB specific params
    parser.add_argument(
        "--method",
        type=str,
        default="2",
        help="gfn method to use",
    )
    parser.add_argument("--bond_opt", action="store_true")
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
        "--average_size",
        type=int,
        default=12,
        help="Average number of atoms resulting from crossover",
    )
    parser.add_argument(
        "--size_stdev",
        type=int,
        default="3",
        help="STD of crossover molecule size distribution",
    )
    return parser.parse_args(arg_list)


def main():
    """Main function that starts the GA."""
    args = get_arguments()

    # Get arguments as dict and add scoring function to dict.
    args_dict = vars(args)

    # Create list of dicts for the distributed GAs
    GA_args = args_dict

    # Variables for crossover module
    co.average_size = args.average_size
    co.size_stdev = args.size_stdev

    # Run the GA
    for i in range(args.n_tries):
        # Start the time
        t0 = time.time()
        # Create output_dir
        GA_args["output_dir"] = args_dict["output_dir"] / f"gen_{i}"
        Path(GA_args["output_dir"]).mkdir(parents=True, exist_ok=True)

        # Setup logger
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(GA_args["output_dir"], "printlog.txt"), mode="w"
                ),
                logging.StreamHandler(),  # For debugging. Can be removed on remote
            ],
        )

        # Log current git commit hash
        logging.info("Current git hash: %s", get_git_revision_short_hash())

        # Log the argparse set values
        logging.info("Input args: %r", args)
        GA = GeneticAlgorithm(GA_args)
        GA.run()

        # Final output handling and logging
        t1 = time.time()
        logging.info(f"# Total duration: {(t1 - t0) / 60.0:.2f} minutes")

    # Ensure the program exists when running on the frontend.
    sys.exit(0)


if __name__ == "__main__":
    main()
