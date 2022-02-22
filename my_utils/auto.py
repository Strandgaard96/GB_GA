import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from glob import glob
import pickle
import shutil
import subprocess
from ase.io import read, write, Trajectory
from ase.visualize import view


def shell(cmd, shell=False):
    """
    Subprocess handler function
    Args:
        cmd (str): String to pass to bash shell
        shell (bool): Specifies whether run as bash shell or not

    Returns:
        output (str): Program output
        err (str): Possible error messages
    """

    if shell:
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    else:
        cmd = cmd.split()
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    output, err = p.communicate()
    return output, err


def _plot_energy(energy):
    """Helper method for plotting enregies"""
    # Convert to eV from hartree
    font = {"size": 18}

    plt.rcParams["figure.figsize"] = (13, 6)
    plt.matplotlib.rc("font", **font)
    energy = [_ * 27.2114 for _ in energy]
    plt.plot(np.array(energy), "-*")
    plt.xlabel("Steps")
    plt.ylabel("Energy(eV)")
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    plt.show()


def plt_handler():

    base = Path("/home/magstr/Documents/nitrogenase/diagrams/")

    # Get folder paths
    paths = []
    for root, dirs, files in os.walk(base / "new_opt"):
        for file in files:
            if file.endswith("xtbopt.log"):
                paths.append(Path(os.path.join(root, file)))

    for elem in paths:
        print(elem)
        logfile = elem
        energy = extract_energyxtb(logfile)
        _plot_energy(energy)


def see_structures():
    """Visualize xyz structures in folder"""
    paths = []
    struct = ".xyz"
    dest = "/home/magstr/Documents/GB_GA/test/Runs_cycle"
    for root, _, files in os.walk(dest):
        for file in files:
            if file.endswith(struct):
                paths.append(os.path.join(root, file))
    for elem in paths:
        print(elem)
        #shutil.copyfile(elem, elem[0 : -len(struct)] + "traj.xyz")
        atoms = read(elem, index=":")
        view(atoms, block=True)


def get_paths_custom(source, struct, dest):

    print("getting paths")
    paths = []

    shutil.copytree(source, dest, dirs_exist_ok=True)
    for root, dirs, files in os.walk(dest):
        for file in files:
            if file.endswith(struct):
                paths.append(Path(os.path.join(root, file)))
    return paths


if __name__ == "__main__":
    # plt_handler()
    # get_indices_cofactor()
    # process_crds()
    see_structures()
