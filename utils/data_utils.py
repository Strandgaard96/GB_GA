# improts
import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# For highlight colors
from rdkit import Chem

from my_utils.constants import kcal

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))

from ppqm.xtb import read_properties
from support_mvp.backup_plot_diagram.data_handler import (
    read_energy_opt_orca,
    read_parts,
    read_properties_sp,
)

HARTREE2KCAL = 627.51
kcal = 627.51

# Custom functions
from my_utils.classes import Conformers, Individual


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


def rdkit_embed_scoring_calc_opt(arr):
    N2_NH3 = arr["Mo_N2_NH3"]
    NH3 = arr["Mo_NH3"]
    delta = N2_NH3 * kcal - (NH3 * kcal + parts_dict["N2"])
    return delta


def rdkit_embed_scoring_NH3toN2_calc_opt(arr):
    N2 = arr["Mo_N2"]
    NH3 = arr["Mo_NH3"]
    delta = (N2 * kcal + parts_dict["NH3"]) - (NH3 * kcal + parts_dict["N2"])
    return delta


def rdkit_embed_scoring_NH3plustoNH3_calc_opt(arr):
    NH3plus = arr["Mo_NH3plus"]
    NH3 = arr["Mo_NH3"]
    delta = NH3 * kcal - NH3plus * kcal + parts_dict["delta_Cp"]
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


def get_fully_cycle_opt_structures():

    folder = Path(
        "/home/magstr/Documents/schrock/diagrams_schrock/dft/candidate_cycle/"
    )
    f = sorted(folder.glob("dft_folder*/*"))

    ind_list = []

    conformer_file = Path("/home/magstr/Documents/GB_GA/data/final_dft_opt.pkl")
    with open(conformer_file, "rb") as file:
        conf = pickle.load(file)
    conf.sortby("final_dft_opt")

    smiles = [
        "CCCN1CCCC(N)(C2CCCC2)C1",
        "CCCOC(=O)NCC1(N)CCCCC1",
        "NC1(CCCCCl)CCCCC1",
        "CC(C)(N)CCC1CCCCC1",
        "CC(N)Cc1ccc(Cc2ccccc2)cc1",
        "CC(N)Cc1ccc(CCl)cc1",
        "CC(=O)Nc1ccc(F)c(C(=O)Cc2ccnc(N)c2)c1",
        "C=C(C)CC(C)(C)N",
        "CC(=O)c1ccc(F)c(C(=O)CN)c1",
        "CC(=O)c1ccc(F)c(C(=O)N=CN)c1",
        "CC(=O)c1cc(O)c(C(=O)CN)cc1O",
        "N#Cc1ccnc(C#N)c1N",
        "N#Cc1ccnc(C(=O)Cl)c1N",
        "N#Cc1cnc(C#N)c(N)c1",
        "N#Cc1ncc(CC(=O)O)cc1N",
        "N#Cc1ncc(C(=O)CO)cc1N",
        "Nc1ccccc1N=CC(=O)Cl",
        "NCCc1ncnc2ccccc12",
        "NCC=Cc1ncnc2ccccc12",
        "NC=NC(=O)c1cccc(Br)c1",
        "NCCCc1ncnc2ccccc12",
        "NCCCOc1ccnc2cccnc12",
        "NCCCOc1ncnc2cccnc12",
        "NCCOC(=O)c1ncnc2ccccc12",
    ]
    for elem in f:

        if elem.name in smiles:
            converged = True
        else:
            continue

        for mol in conf.molecules:
            if mol.smiles == elem.name:
                ind = mol
                break

        energy_dict = {}
        struct_dict = {}
        files = sorted(elem.rglob("orca.out"))
        scoring = ind.scoring_function

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

        delta = funcs[scoring](energy_dict)

        ind = copy.deepcopy(mol)
        setattr(ind, f"reaction_profile_pbe", delta)
        setattr(ind, f"reaction_profile_final_structs", struct_dict)

        ind_list.append(ind)

    with open("data/converged_reaction_profiles.pkl", "wb") as output:
        pickle.dump(ind_list, output, pickle.HIGHEST_PROTOCOL)

    print("lol")


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
    dicts = read_parts()

    parts_dict = dicts["parts_orca"]
    # Scoring function dict
    funcs = {
        "rdkit_embed_scoring": rdkit_embed_scoring_calc_opt,
        "rdkit_embed_scoring_NH3toN2": rdkit_embed_scoring_NH3toN2_calc_opt,
        "rdkit_embed_scoring_NH3plustoNH3": rdkit_embed_scoring_NH3plustoNH3_calc_opt,
    }

    # with open(f"test.pkl", "rb") as f:
    #    conf = pickle.load(f)
    # with open(f"data/final_dft_opt.pkl", "rb") as f:
    #    conf = pickle.load(f)
    get_fully_cycle_opt_structures()

    print("Donzo")
    # get_opts()
    # rename()
    # main()
