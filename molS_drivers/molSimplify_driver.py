"""

This is the driver script for doing various molS operations on the Mo catalyst core

Example:
    How to run:

        $ python molSymplify_driver.py args

Todo:
    * Find way of getting attom idx for smicat
"""
import sys, os
from pathlib import Path
import json, shutil, argparse, glob, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_utils.my_utils import cd
from my_utils.my_xtb_utils import run_xtb
from my_utils.auto import shell, get_paths_custom, get_paths_molsimplify


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
        type=str,
        default="Runs",
        help="Sets the output dir of the molSimplify commands",
    )
    parser.add_argument(
        "--cycle_dir",
        type=str,
        default="Runs/Runs_cycle",
        help="Sets the output dir of the cycle molSimplify commands",
    )
    parser.add_argument(
        "--cleanup",
        dest="cleanup",
        action="store_true",
        help="Cleans xtb output folders",
    )
    parser.set_defaults(cleanup=True)
    parser.add_argument(
        "--ligand_smi",
        type=str,
        default="[1H]C(=O)[C@H](Oc1ccc(-c2ccc(C#N)cc2)cc1)c1ccccc1",
        help="Set the ligand to put on the core",
    )
    parser.add_argument(
        "--xtbout",
        type=str,
        default="xtbout",
        help="Directory to put xtb calculations",
    )
    return parser.parse_args(arg_list)


def create_cycleMS(new_core=None, smi_path=None, run_dir=None):
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

    for key, value in smi_dict.items():

        intermediate_cmd = (
            f"molsimplify -core {new_core} -lig {value['smi']} -ligocc 1"
            f" -ccatoms 8 -skipANN True -spin 1 -oxstate 3 -ffoption no"
            f" -smicat 1 -suff intermediate_{key} -rundir {run_dir}"
        )

        # Modify string if multiple ligands.
        # Which is the case for equatorial NN at Mo_N2_NH3.
        if key == "Mo_N2_NH3":
            intermediate_cmd = None
            intermediate_cmd = (
                f"molsimplify -core {new_core} -oxstate 3 -lig [NH3],N#N"
                f" -ligocc 1,1 -ligloc True -ccatoms 8,8 -spin 1 -oxstate 3"
                f" -skipANN True -smicat [1],[1]"
                f" -ffoption no -suff intermediate_{key} -rundir {run_dir}"
            )

        print(f"String passed to shell: {intermediate_cmd}")
        out, err = shell(
            intermediate_cmd,
            shell=False,
        )
        with open(f"job_{key}.out", "w", encoding="utf-8") as f:
            f.write(out)
        with open(f"err_{key}.out", "w", encoding="utf-8") as f:
            f.write(err)

    # Finaly create directory that contains the simple core
    # (for consistentxtb calcs)
    bare_core_dir = Path(run_dir) / "intermediate_Mo" / "struct"
    bare_core_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(new_core, bare_core_dir)

    return


def xtb_calc(cycle_dir=None, param_path=None, dest="xtbout"):
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
            run_xtb(
                structure=elem.name,
                method="gfn2",
                type="ohess",
                charge=charge,
                spin=spin,
                gbsa="Benzene",
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
    for file in glob.glob("*.out"):
        shutil.move(file, dest)

    log_path = dest / "logfiles"
    os.mkdir(log_path)

    logfiles = dest.rglob("*xtbjob.out")
    for file in logfiles:
        folder = log_path / file.parents[0].name
        folder.mkdir(exist_ok=True)
        shutil.copy(str(file), str(folder))
        os.rename(folder / "xtbjob.out", folder / "job.out")
    try:
        os.remove("new_core.xyz")
        os.remove("CLIinput.inp")
    except OSError:
        pass

    return None


def create_custom_core(args):
    """

    Create a custom core based on a proposed ligand.

    Args:
        args (Namespace): args from driver

    Returns:

    """
    lig_smi = args.ligand_smi
    core_file = Path("../templates/core_withHS.xyz")
    replig = 1
    suffix = "core_custom"
    run_dir = Path(args.run_dir)
    smicat = "TODO"  # TODO get idx

    core_cmd = (
        f"molsimplify -core {core_file} -lig {lig_smi} -replig {replig} -ligocc 3"
        f" -ccatoms 24, 25, 26 -skipANN True -spin 1 -oxstate 3 -ffoption After"
        f" -coord 5 -keepHs no -smicat 2 -rundir {run_dir} -suff {suffix}"
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
    run_dir = Path(args.run_dir)

    # What ligand do i want to molS_drivers?
    create_custom_core(args)

    # Check for output structure
    new_core = sorted(run_dir.glob("**/*.xyz"))
    intermediate_smi_path = Path("../data/intermediate_smiles.json")

    print(new_core)

    if new_core:
        # Move output structure to current dir
        shutil.copy(new_core[0], "./new_core.xyz")
        # Pass the output structure to cycle creation
        create_cycleMS(
            new_core="new_core.xyz",
            smi_path=intermediate_smi_path,
            run_dir=args.cycle_dir,
        )

    # Create xtb outpout folder
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(args.xtbout) / (args.ligand_smi + timestr)
    dest.mkdir(parents=True, exist_ok=False)
    xtb_calc(cycle_dir=args.cycle_dir, param_path=intermediate_smi_path, dest=dest)

    # for debug
    # os.mkdir(str(dest / "Mo"))
    # with open(dest / "Mo" / "xtbjob.out", "w") as f:
    #    f.write("test")

    if args.cleanup:
        collect_logfiles(dest=dest)


if __name__ == "__main__":
    intermediate_smi_path = Path("../data/intermediate_smiles.json")
    cycle_dir = "Runs/Runs_cycle"
    create_cycleMS(
        new_core="../templates/core_withHS.xyz",
        smi_path=intermediate_smi_path,
        run_dir=cycle_dir,
    )
    # main()
