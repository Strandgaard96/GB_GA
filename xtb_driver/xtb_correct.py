import concurrent.futures
import json
import os
from pathlib import Path

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import argparse
import concurrent.futures
import os
import random
import string
import subprocess
from datetime import datetime
from pathlib import Path

import submitit

# Get dict with intermediate variables
with open(source / "data/intermediate_smiles.json", "r", encoding="utf-8") as f:
    smi_dict = json.load(f)


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run xtb calcs", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=6,
        help="Number of cores to distribute over",
    )
    parser.add_argument(
        "--calc_dir",
        type=Path,
        default=".",
        help="Path to folders",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=6,
        help="How many cores for each calc",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=1,
        help="How many GB requested for each calc",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="xtb_debug",
        help="Directory to put various files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=12,
        help="Minutes before timeout",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--hess", action="store_true")
    # XTB specific params
    parser.add_argument(
        "--method",
        type=str,
        default="gfn 2",
        help="gfn method to use",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="xtbopt",
        help="",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="kemi1",
        help="",
    )
    parser.add_argument(
        "--opt_type",
        dest="opt_type",
        choices=["opt", "ohess", "hess"],
        required=True,
        help="""""",
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
    return parser.parse_args(arg_list)


def read_results(output, err):
    if not "normal termination" in err:
        return {"atoms": None, "coords": None, "energy": None}
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
    return {"atoms": atoms, "coords": coords, "energy": energy}


def run_xtb(args):
    """Submit xtb calculations with given params.

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
        output, err = popen.communicate(timeout=120 * 60)
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


class XTB_optimizer:
    """Base XTB optimizer class."""

    def __init__(self):

        # Initialize default xtb values
        self.method = "gfnff"
        self.workers = 1
        # xtb runner function
        self.xtb_runner = run_xtb
        # xtb options
        self.XTB_OPTIONS = {}

        # Start with ff optimization
        cmd = f"xtb --{self.method}"
        for key, value in self.XTB_OPTIONS.items():
            cmd += f" --{key} {value}"
        self.cmd = cmd

    def add_options_to_cmd(self, option_dict):
        """From passed dict get xtb options if it has the appropriate keys and
        add to xtb string command."""

        option_dict["charge"] = smi_dict["Mo_N2_NH3"]["charge"]
        option_dict["uhf"] = smi_dict["Mo_N2_NH3"]["spin"]

        # XTB options to check for
        options = ["gbsa", "spin", "charge", "uhf", "input"]

        # Get commandline options
        commands = {k: v for k, v in option_dict.items() if k in options}
        for key, value in commands.items():
            self.cmd += f" --{key} {value}"

        # Set opt type
        self.cmd += f" --{option_dict['opt_type']}"

    def optimize(self, args):
        """Do paralell optimization of all the entries in args."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers
        ) as executor:
            results = executor.map(self.xtb_runner, args)
        return results


class XTB_optimize_file(XTB_optimizer):
    """Specific xtb optimizer class for the schrock intermediates."""

    def __init__(self, file, scoring_options, **kwargs):
        """

        Args:
            file Path: XYZ structure to score
            scoring_options (dict): Scoring options for xtb
        """

        # Inherit the basic xtb functionality from XTB_OPTIMIZER
        super().__init__(**kwargs)

        # Set class attributes
        self.file = file
        self.options = scoring_options

        # Change from ff to given method
        self.cmd = self.cmd.replace("gfnff", f"{self.options['method']}")

        # Set additional xtb options
        self.add_options_to_cmd(self.options)

        # Set folder name if given options dict
        if not "name" in self.options:
            self.name = "tmp_" + "".join(
                random.choices(string.ascii_uppercase + string.digits, k=4)
            )
        else:
            self.name = self.options["name"]

        # set SCRATCH if environmental variable
        try:
            self.scr_dir = os.environ["SCRATCH"]
        except:
            self.scr_dir = os.getcwd()
        print(f"SCRATCH DIR = {self.scr_dir}")

    def optimize_file(self):
        """Optimize the given xyz file."""

        # Set paralellization options
        cpus_per_worker = self.options["cpus_per_task"]

        # Create args tuple and submit ff calculation
        args = (self.file, self.cmd, cpus_per_worker, self.file.parent, "gfn2")

        results = self.xtb_runner(args)

        return results


def slurm_xtb(sc_function, scoring_args_list):

    scoring_args = scoring_args_list[0][1]

    executor = submitit.AutoExecutor(
        folder=scoring_args["output_dir"] / "scoring_tmp",
        slurm_max_num_timeout=0,
    )
    mem_per_cpu = (scoring_args["memory"] * 1000) // scoring_args["cpus_per_task"]
    executor.update_parameters(
        name=f"conformer_search",
        cpus_per_task=scoring_args["cpus_per_task"],
        slurm_mem_per_cpu=f"{mem_per_cpu}MB",
        timeout_min=scoring_args["timeout"],
        slurm_partition=scoring_args["partition"],
        slurm_array_parallelism=20,
    )

    jobs = executor.map_array(
        sc_function,
        scoring_args_list,
    )

    results = [catch(job.result, handle=lambda e: (None, None)) for job in jobs]

    return results


def catch(func, *args, handle=lambda e: e, **kwargs):
    """Helper function that takes the submitit result and returns an exception
    if no results can be retrieved."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)


def score_func(args):

    file, scoring_args = args

    # Get the key for the current structure
    key = str(file.parent.name)
    scoring_args["key"] = key

    optimizer = XTB_optimize_file(file=file, scoring_options=scoring_args)

    # Perform calculation
    result = optimizer.optimize_file()

    return result


def xtb_folder_driver():

    args = get_arguments()

    p = args.calc_dir
    paths = sorted(p.rglob(f"{args.name}.xyz"))

    sub_args = [(file, vars(args)) for file in paths]

    # Test function first
    # score_func(sub_args[0])

    results = slurm_xtb(score_func, sub_args)

    print("Finished calcs")

    return


if __name__ == "__main__":
    xtb_folder_driver()
