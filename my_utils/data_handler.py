# improts
import io
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# For highlight colors
from rdkit import Chem


source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))

# Custom functions
from my_utils.classes import Conformers, Individual

HARTREE2EV = 27.2114
HARTREE2KCAL = 627.51
kcal = 627.51
ev = 27.2114

hartree2kcalmol = 627.5094740631
CORE_ELECTRONIC_ENERGY = -32.698
kcaltohartree = 1 / 627.5

NH3_ENERGY_gfn2 = -4.427496335658
N2_ENERGY_gfn2 = -5.766345142003
CP_RED_ENERGY_gfn2 = 0.2788559959203811

NH3_ENERGY_gfn1 = -4.834742774551
N2_ENERGY_gfn1 = -6.331044264474
CP_RED_ENERGY_gfn1 = 0.2390159933706209

GAS_ENERGIES = {
    "2": (NH3_ENERGY_gfn2, N2_ENERGY_gfn2, CP_RED_ENERGY_gfn2),
    "1": (NH3_ENERGY_gfn1, N2_ENERGY_gfn1, CP_RED_ENERGY_gfn1),
}


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


def get_energy_dicts():
    path = Path(
        "/home/magstr/Documents/nitrogenase/niflheim_scripts/nitrogenase/reference_energies"
    )
    dict1_path = path / "gfn1_dict.json"
    dict2_path = path / "gfn2_dict.json"
    dict3_path = path / "dft_ams_dict.json"
    dict4_path = path / "dft_orca_dict.json"
    dict5_path = path / "dft_orca_dict_svp.json"
    dict6_path = path / "dft_orca_sarcJ_dict.json"

    # Check for existing reference energy_dicts for speedup
    if (
        (dict1_path.exists())
        and (dict2_path.exists())
        and (dict3_path.exists())
        and (dict4_path.exists())
        and (dict5_path.exists())
        and (dict6_path.exists())
    ):
        with open(dict1_path, "r", encoding="utf-8") as f:
            gfn1_dict = json.load(f)
        with open(dict2_path, "r", encoding="utf-8") as f:
            gfn2_dict = json.load(f)
        with open(dict3_path, "r", encoding="utf-8") as f:
            dft_ams_dict = json.load(f)
        with open(dict4_path, "r", encoding="utf-8") as f:
            dft_orca_dict = json.load(f)
        with open(dict5_path, "r", encoding="utf-8") as f:
            dft_orca_svp_dict = json.load(f)
        with open(dict6_path, "r", encoding="utf-8") as f:
            dft_orca_sarcJ_dict = json.load(f)

        return (
            gfn1_dict,
            gfn2_dict,
            dft_ams_dict,
            dft_orca_dict,
            dft_orca_svp_dict,
            dft_orca_sarcJ_dict,
        )

    # Correction values
    g1_lut_corr = 10
    g1_cp_corr = -65

    g2_lut_corr = 0
    g2_cp_corr = -40

    gfn1_folders = (path / "gas_gfn1").rglob("*job.out")
    gfn2_folders = (path / "gas_gfn2").rglob("*job.out")

    gfn1_dict = {}
    for elem in gfn1_folders:
        gfn1_dict[elem.parent.name] = (
            HARTREE2KCAL * get_energies(paths=[elem], pattern="TOTAL ENERGY")[0]
        )
    gfn1_dict["delta_Cp"] = gfn1_dict["CrCp2+"] - gfn1_dict["CrCp2"] + g1_cp_corr
    gfn1_dict["delta_Lu"] = gfn1_dict["Lu"] - gfn1_dict["LuH+"] + g1_lut_corr

    gfn2_dict = {}
    for elem in gfn2_folders:
        gfn2_dict[elem.parent.name] = (
            HARTREE2KCAL * get_energies(paths=[elem], pattern="TOTAL ENERGY")[0]
        )
    gfn2_dict["delta_Cp"] = gfn2_dict["CrCp2+"] - gfn2_dict["CrCp2"] + g2_cp_corr
    gfn2_dict["delta_Lu"] = gfn2_dict["Lu"] - gfn2_dict["LuH+"] + g2_lut_corr

    dft_ams_folders = (path / "schrock_parts_ams").rglob("*dft.out")
    dft_ams_dict = {}
    for elem in dft_ams_folders:
        dft_ams_dict[elem.parent.name] = (
            HARTREE2KCAL * get_energies(paths=[elem], pattern="current energy")[0]
        )
    dft_ams_dict["delta_Cp"] = dft_ams_dict["CrCp2+"] - dft_ams_dict["CrCp2"]
    dft_ams_dict["delta_Lu"] = dft_ams_dict["Lu"] - dft_ams_dict["LuH+"]

    dft_orca_folders = (path / "parts_orca").rglob("*orca.out")
    dft_orca_dict = {}
    for elem in dft_orca_folders:
        dft_orca_dict[elem.parent.name] = (
            HARTREE2KCAL * get_energies(paths=[elem], pattern="FINAL SINGLE POINT")[0]
        )
    dft_orca_dict["delta_Cp"] = dft_orca_dict["CrCp2+"] - dft_orca_dict["CrCp2"]
    dft_orca_dict["delta_Lu"] = dft_orca_dict["Lu"] - dft_orca_dict["LuH+"]

    dft_orca_svp_folders = (path / "parts_orca_svp").rglob("*orca.out")
    dft_orca_svp_dict = {}
    for elem in dft_orca_svp_folders:
        dft_orca_svp_dict[elem.parent.name] = (
            HARTREE2KCAL * get_energies(paths=[elem], pattern="FINAL SINGLE POINT")[0]
        )
    dft_orca_svp_dict["delta_Cp"] = (
        dft_orca_svp_dict["CrCp2+"] - dft_orca_svp_dict["CrCp2"]
    )
    dft_orca_svp_dict["delta_Lu"] = dft_orca_svp_dict["Lu"] - dft_orca_svp_dict["LuH+"]

    dft_orca_sarcJ_folders = (path / "parts_orca_nodef2").rglob("*orca.out")
    dft_orca_sarcJ_dict = {}
    for elem in dft_orca_sarcJ_folders:
        dft_orca_sarcJ_dict[elem.parent.name] = (
            HARTREE2KCAL * get_energies(paths=[elem], pattern="FINAL SINGLE POINT")[0]
        )
    dft_orca_sarcJ_dict["delta_Cp"] = (
        dft_orca_sarcJ_dict["CrCp2+"] - dft_orca_sarcJ_dict["CrCp2"]
    )
    dft_orca_sarcJ_dict["delta_Lu"] = (
        dft_orca_sarcJ_dict["Lu"] - dft_orca_sarcJ_dict["LuH+"]
    )

    with open(dict1_path, "w") as fp:
        json.dump(gfn1_dict, fp)
    with open(dict2_path, "w") as fp:
        json.dump(gfn2_dict, fp)
    with open(dict3_path, "w") as fp:
        json.dump(dft_ams_dict, fp)
    with open(dict4_path, "w") as fp:
        json.dump(dft_orca_dict, fp)
    with open(dict5_path, "w") as fp:
        json.dump(dft_orca_svp_dict, fp)
    with open(dict6_path, "w") as fp:
        json.dump(dft_orca_sarcJ_dict, fp)

    return (
        gfn1_dict,
        gfn2_dict,
        dft_ams_dict,
        dft_orca_dict,
        dft_orca_svp_dict,
        dft_orca_sarcJ_dict,
    )


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


def rdkit_embed_scoring_calc(N2_NH3, NH3):
    delta = N2_NH3 * kcal - (NH3 * kcal + reactions_dft_orca_sarcJ_tzp["N2"])
    return delta


def rdkit_embed_scoring_NH3toN2_calc(N2, NH3):

    delta = (N2 * kcal + reactions_dft_orca_sarcJ_tzp["NH3"]) - (
        NH3 * kcal + reactions_dft_orca_sarcJ_tzp["N2"]
    )
    return delta


def rdkit_embed_scoring_NH3plustoNH3_calc(NH3plus, NH3):
    delta = NH3 * kcal - NH3plus * kcal + reactions_dft_orca_sarcJ_tzp["delta_Cp"]
    return delta


def get_opts():

    folder = Path("/home/magstr/dft_data/top10_opts/logfiles")
    f = sorted(folder.glob("*"))

    deltas = []
    score_list = []
    for elem in f:
        result_dict = {}
        files = sorted(elem.rglob("orca.out"))
        keys = set([p.parent.name for p in sorted(elem.rglob("*orca.out"))])

        scoring = scoring_from_keys(keys)

        for file in files:
            name = file.parent.name
            en = read_energy_opt_orca(file)
            result_dict[name] = en[-1]

        delta = funcs[scoring](result_dict)
        deltas.append(delta)
        score_list.append(scoring)
    print(f"Scores for top ten conformers {deltas}")
    print(score_list)


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


if __name__ == "__main__":

    # Get the reference energy dicts.
    (
        reactions_gfn1_corrected,
        reactions_gfn2_corrected,
        reactions_dft_ams_tzp,
        reactions_dft_orca_tzp,
        reactions_dft_orca_svp_tzp,
        reactions_dft_orca_sarcJ_tzp,
    ) = get_energy_dicts()

    # Scoring function dict
    funcs = {
        "rdkit_embed_scoring": rdkit_embed_scoring_calc,
        "rdkit_embed_scoring_NH3toN2": rdkit_embed_scoring_NH3toN2_calc,
        "rdkit_embed_scoring_NH3plustoNH3": rdkit_embed_scoring_NH3plustoNH3_calc,
    }
    get_opts()
    # rename()
    # main()
