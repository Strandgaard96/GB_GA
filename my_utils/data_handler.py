# improts
import io
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# For highlight colors
from rdkit import Chem

from my_utils.constants import HARTREE2EV, kcal

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))

from ppqm.xtb import read_properties
from support_mvp.backup_plot_diagram.data_handler import get_energy_dicts

# Custom functions
from my_utils.classes import Conformers, Individual


# Crucial code for unpickling objects with old module names
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "my_utils.my_utils":
            renamed_module = "my_utils.classes"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


def get_zpe(paths=None, pattern="Zero-point"):

    regex = "[\\d]*[.][\\d]+"
    energy = {}
    for elem in paths:
        name = "NEEDS FIXING"
        with (open(elem, "r")) as file:
            for line in file:
                if re.search(pattern, line):
                    num = re.findall(regex, line)
                    energy[name] = float(num[0]) * HARTREE2EV
    return energy


def extract_energyxtb(logfile=None):
    """Extracts xtb energies from xtb logfile using regex matching.

    Args:
        logfile (str): Specifies logfile to pull energy from

    Returns:
        energy (List[float]): List of floats containing the energy in each step
    """

    re_energy = re.compile("energy: (-\\d+\\.\\d+)")
    energy = []
    with logfile.open() as f:
        for line in f:
            if "energy" in line:
                energy.append(float(re_energy.search(line).groups()[0]))
    return energy


def get_energies_Mo(paths=None, pattern="total energy"):
    regex = "-[\\d]*[.][\\d]+"
    energy = {}
    for elem in paths:
        r = re.compile("Mo*")
        name = str(elem).split("/")
        name = list(filter(r.match, name))[0]
        print(name)
        with (open(elem, "r", encoding="utf8")) as file:
            file.seek(0, os.SEEK_END)  # seek to end of file; f.seek(0, 2) is legal
            file.seek(file.tell() - 100 * 1024, os.SEEK_SET)  # go backwards 3 bytes
            for line in file:
                if re.search(pattern, line):
                    num = re.findall(regex, line)
                    energy[name] = float(num[0])
    return energy


def read_properties_sp(logfile):
    """Read Singlepoint Energy."""
    with logfile.open() as f:

        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                scf_energy = float(line.split()[4])
                break
            else:
                scf_energy = np.nan

    return scf_energy


def get_energies(paths=None, pattern="total energy"):
    regex = "-[\\d]*[.][\\d]+"
    energy = []
    for elem in paths:
        with (open(elem, "r", encoding="utf8")) as file:
            for line in file:
                if re.search(pattern, line):
                    num = re.findall(regex, line)
                    energy.append(float(num[0]))
    return energy


def get_paths(endswith="xtbopt.xyz", middle=None):

    paths = []
    for root, dirs, files in os.walk(base / middle):
        for file in files:
            if file.endswith(".out"):
                paths.append(Path(root))
    return paths


def extract_scoring(mol_path, reverse=False, scoring=None, keys=None):

    min_paths = []

    result_dict = {}
    try:
        for key in keys:
            p = mol_path / key
            p = sorted(p.rglob("*orca.out"))

            res = []
            for elem in p:
                res.append(read_properties_sp(elem))
            result_dict[key] = np.array(res)
            min_paths.append(p[np.nanargmin(res)].parent)

        ind_paths = sorted(mol_path.rglob("*ind.pkl"))
        delta = funcs[scoring](result_dict)

        # Get the ind object
        ind_path = ind_paths[0]
        with open(ind_path, "rb") as f:
            ind = pickle.load(f)
    except Exception as e:
        print(e)
        print(f"Something failed for {mol_path}")
        delta = np.nan
        ind = Individual(rdkit_mol=Chem.MolFromSmiles("CN"))
        min_paths = [(0, 0)]

    return ind, delta, min_paths


def read_energy_opt_orca(logfile=None):
    """Read the energies of ORCA optimization file."""
    re_energy = re.compile("(-\\d+\\.\\d+)")
    energy = []
    with logfile.open() as f:
        for line in f:
            if "FINAL SINGLE POINT" in line:
                energy.append(float(re_energy.search(line).groups()[0]))
    return energy


def rdkit_embed_scoring_calc(Mo_N2_NH3, Mo_NH3):
    delta = Mo_N2_NH3 * kcal - (Mo_NH3 * kcal + reactions_dft_orca_sarcJ_tzp["N2"])
    return delta


def rdkit_embed_scoring_NH3toN2_calc(Mo_N2, Mo_NH3):

    delta = (Mo_N2 * kcal + reactions_dft_orca_sarcJ_tzp["NH3"]) - (
        Mo_NH3 * kcal + reactions_dft_orca_sarcJ_tzp["N2"]
    )
    return delta


def rdkit_embed_scoring_NH3plustoNH3_calc(Mo_NH3plus, Mo_NH3):
    delta = Mo_NH3 * kcal - Mo_NH3plus * kcal + reactions_dft_orca_sarcJ_tzp["delta_Cp"]
    return delta


def get_opts():

    folder = Path("/home/magstr/dft_data/0_140_dft_opts")
    f = sorted(folder.glob("*/*"))

    ind_list = []

    for elem in f:
        energy_dict = {}
        struct_dict = {}
        files = sorted(elem.rglob("orca.out"))
        keys = set([p.parent.name for p in sorted(elem.rglob("*orca.out"))])
        scoring = scoring_from_keys(keys)

        for file in files:
            name = file.parent.name
            en = read_energy_opt_orca(file)

            with open(file.parent / "orca.xyz", "r") as f:
                lines = f.readlines()

            # Get the optimized struct
            struct_dict[name] = lines
            # Get the last energy
            energy_dict[name] = en[-1]

        if "Mo_NH3+" in energy_dict.keys():
            energy_dict["Mo_NH3plus"] = energy_dict.pop("Mo_NH3+", "Mo_NH3plus")

        delta = funcs[scoring](**energy_dict)

        # Add the results dict to the ind object
        ind_paths = sorted(elem.rglob("*ind.pkl"))
        # Get the ind object
        ind_path = ind_paths[0]
        with open(ind_path, "rb") as f:
            ind = pickle.load(f)
        setattr(ind, f"final_dft_opt", delta)
        setattr(ind, f"final_structs", struct_dict)
        ind_list.append(ind)

    final_ind_list = [ind for ind in ind_list if ind.final_dft_opt > -100]
    # Save the final object for post-processing and analysis!
    conf = Conformers(molecules=final_ind_list)
    # Sort before saving
    conf.sortby("final_dft_opt")
    conf.save(directory="data", name="final_dft_opt.pkl")


def scoring_from_keys(keys):
    if "Mo_NH3" and "Mo_NH3+" in keys:
        scoring = "rdkit_embed_scoring_NH3plustoNH3"
    elif "Mo_NH3" and "Mo_N2" in keys:
        scoring = "rdkit_embed_scoring_NH3toN2"
    elif "Mo_NH3" and "Mo_N2_NH3" in keys:
        scoring = "rdkit_embed_scoring"
    return scoring


def main():
    folder = Path(str(sys.argv[1]))
    f = sorted(folder.glob("*/*"))
    total_inds = []
    for elem in f:

        keys = set([p.parents[1].name for p in sorted(elem.rglob("*orca.out"))])

        scoring = scoring_from_keys(keys)

        print(scoring, elem)
        ind, delta, min_paths = extract_scoring(elem, scoring=scoring, keys=keys)

        if not np.isnan(delta):
            setattr(ind, "dft_singlepoint_conf", delta)
            setattr(ind, "min_confs", min_paths)
            structs = {}
            for dir in min_paths:
                with open(dir / "struct.xyz", "r") as f:
                    lines = f.readlines()
                structs[f"{dir.parent.name}"] = lines
            setattr(ind, f"final_structs", structs)
            total_inds.append(ind)
        else:
            continue

    conf = Conformers(molecules=total_inds)
    # Sort before saving
    conf.sortby("dft_singlepoint_conf")
    conf.save(directory=".", name=f"{str(sys.argv[2])}")


def get_xtb_free_energies(path):
    with (open(path, "r", encoding="utf8")) as file:
        lines = file.readlines()
    props = read_properties(lines, options={"ohess": None})
    return props


if __name__ == "__main__":

    # Get the reference energy dicts.
    (
        reactions_gfn1_corrected,
        reactions_gfn2_corrected,
        reactions_dft_ams_tzp,
        reactions_dft_orca_tzp,
        reactions_dft_orca_svp_tzp,
        reactions_dft_orca_sarcJ_tzp,
        reactions_dft_orca_b3lyp,
        reactions_dft_orca_r2scan,
    ) = get_energy_dicts()

    # Scoring function dict
    funcs = {
        "rdkit_embed_scoring": rdkit_embed_scoring_calc,
        "rdkit_embed_scoring_NH3toN2": rdkit_embed_scoring_NH3toN2_calc,
        "rdkit_embed_scoring_NH3plustoNH3": rdkit_embed_scoring_NH3plustoNH3_calc,
    }
    # with open(f"data/final_dft_opt.pkl", "rb") as f:
    #    conf = pickle.load(f)

    print("Donzo")
    # get_opts()
    # rename()
    # main()
