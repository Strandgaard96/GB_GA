# -*- coding: utf-8 -*-
""" Scoring module
Module handling the driver for scoring ligand candidates.
Contains various global variables that should be available to the scoring
function at all times

"""
import json
import os
import pickle
import sys
from pathlib import Path

from rdkit import Chem

scoring_dir = os.path.dirname(__file__)
sys.path.append(scoring_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from make_structures import (
    connect_ligand,
    connectMols,
    create_dummy_ligand,
    create_prim_amine_revised,
    embed_rdkit,
    mol_with_atom_index,
    remove_N2,
    remove_NH3,
)

from my_utils.classes import Generation, Individual
from my_utils.utils import cd
from my_utils.xtb_class import XTB_optimize_schrock

hartree2kcalmol = 627.5094740631
CORE_ELECTRONIC_ENERGY = -32.698

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


"""int: Module level constants
hartree2kcalmol: Handles conversion from hartree to kcal/mol
CORE_ELECTRONIC_ENERGY: The electronic energy of the Mo core with cut
ligands
NH3_ENERGY_gfn2: Electronic energy of pure NH3, 
used for scoring the NH3 dissacossiation reaction
"""

file = "templates/core_dummy.sdf"
core = Chem.SDMolSupplier(file, removeHs=False, sanitize=False)
"""Mol: 
mol object of the Mo core with dummy atoms instead of ligands
"""
file_NH3 = "templates/core_NH3_dummy.sdf"
core_NH3 = Chem.SDMolSupplier(file_NH3, removeHs=False, sanitize=False)
"""Mol: 
mol object of the Mo core with NH3 in axial position and
dummy atoms instead of ligands
"""

file_N2_NH3 = "templates/core_N2_NH3_dummy.sdf"
core_N2_NH3 = Chem.SDMolSupplier(file_N2_NH3, removeHs=False, sanitize=False)
"""Mol: 
mol object of the Mo core with NH3 in axial position and
dummy atoms instead of ligands
"""


with open("data/intermediate_smiles.json", "r", encoding="utf-8") as f:
    smi_dict = json.load(f)
"""dict: 
Dictionary that contains the smiles string for each N-related intermediate
and the charge and spin for the specific intermediate
"""


def rdkit_embed_scoring(ligand, scoring_args):
    """
    Score the NH3 -> N2_NH3 exchange

    Args:
        ligand (Chem.rdchem.Mol): ligand to put on Mo core
        scoring_args (dict): dict with all relevant args for xtb and general scoring

    Returns:
        De (float): Scoring energy
        Mo_N2_NH3_3d_geom: Geometry of intermediate
        Mo_NH3_3d_geom: Geometry of intermediate
        minidx (int): Index of the lowest energy conformer
    """

    # Get ligand tuple idx.
    idx = ligand.idx

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

    # Get mol object of Mo core + connected ligand
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_N2_NH3 = connect_ligand(core_N2_NH3[0], ligand_cut, NH3_flag=True, N2_flag=True)
    Mo_N2_NH3 = Chem.AddHs(Mo_N2_NH3)

    # Embed mol object
    Mo_N2_NH3_3d = embed_rdkit(
        mol=Mo_N2_NH3, core=core_N2_NH3[0], numConfs=scoring_args["n_confs"], pruneRmsThresh=scoring_args["RMS_thresh"]
    )

    # Go into the output directory
    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_N2_NH3"
        scoring_args["charge"] = smi_dict["Mo_N2_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_N2_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_N2_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        Mo_N2_NH3_energy, Mo_N2_NH3_3d_geom, minidx = optimizer.optimize_schrock()

    # THIS VALUE IS HARDCODED IN xtb_pre_optimize!
    if Mo_N2_NH3_energy == 9999:
        return 9999, None, None, None

    # Remove higher energy conformes from mol object
    discard_conf = [x for x in range(len(Mo_N2_NH3_3d.GetConformers())) if x != minidx]
    for elem in discard_conf:
        Mo_N2_NH3_3d.RemoveConformer(elem)

    # Remove N2 on the full embedding
    Mo_NH3_3d = remove_N2(Mo_N2_NH3_3d)
    Mo_NH3_3d = Chem.AddHs(Mo_NH3_3d)

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        Mo_NH3_energy, Mo_NH3_3d_geom, minidx = optimizer.optimize_schrock()

    if Mo_NH3_energy == 9999:
        return 9999, None, None, None

    # Handle the error and return if xtb did not converge
    if None in (Mo_N2_NH3_energy, Mo_NH3_energy):
        raise Exception(f"None of the XTB calculations converged")

    De = ((Mo_N2_NH3_energy - (Mo_NH3_energy + N2_ENERGY))) * hartree2kcalmol
    print(f"diff energy: {De}")
    return De, Mo_N2_NH3_3d_geom, Mo_NH3_3d_geom, minidx


def rdkit_embed_scoring_NH3toN2(ligand, scoring_args):
    """
    Score the NH3 -> N2 exchange

        Args:
            ligand (Chem.rdchem.Mol): ligand to put on Mo core
            scoring_args (dict): dict with all relevant args for xtb and general scoring

        Returns:
            De (float): Scoring energy
            Mo_N2_3d_geom: Geometry of intermediate
            Mo_NH3_3d_geom: Geometry of intermediate
            minidx (int): Index of the lowest energy conformer
    """

    # Get tuple idx
    idx = ligand.idx

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

    # Get mol object of Mo core + connected ligand
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_NH3 = connect_ligand(core_NH3[0], ligand_cut, NH3_flag=True)
    Mo_NH3 = Chem.AddHs(Mo_NH3)

    # Embed mol object
    Mo_NH3_3d = embed_rdkit(
        mol=Mo_NH3,
        core=core_NH3[0],
        numConfs=scoring_args["n_confs"],
    )

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        Mo_NH3_energy, Mo_NH3_3d_geom, minidx = optimizer.optimize_schrock()

    # THIS VALUE IS HARDCODED IN xtb_pre_optimize!
    if Mo_NH3_energy == 9999:
        return 9999, None, None, None

    # Remove higher energy conformers from mol object
    discard_conf = [x for x in range(len(Mo_NH3_3d.GetConformers())) if x != minidx]
    for elem in discard_conf:
        Mo_NH3_3d.RemoveConformer(elem)

    # Replace NH3 with N2
    Mo_N2 = Chem.ReplaceSubstructs(
        Mo_NH3_3d,
        Chem.AddHs(Chem.MolFromSmarts("[NH3]")),
        Chem.MolFromSmarts("N#N"),
        replaceAll=True,
    )[0]

    # Get bare Mo to use as embed refference
    Mo_3d = remove_NH3(Mo_NH3_3d)
    Mo_3d = Chem.AddHs(Mo_3d)

    # Change charge of the N bound to Mo to ensure sanitation works
    match = Mo_N2.GetSubstructMatch(Chem.MolFromSmarts("[Mo]N#N"))
    Mo_N2.GetAtomWithIdx(match[1]).SetFormalCharge(1)

    # Embed catalyst
    Mo_N2_3d = embed_rdkit(
        mol=Mo_N2,
        core=Mo_3d,
        numConfs=1,
    )

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_N2"
        scoring_args["charge"] = smi_dict["Mo_N2"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_N2"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_N2_3d, scoring_options=scoring_args)

        # Perform calculation
        Mo_N2_energy, Mo_N2_3d_geom, minidx = optimizer.optimize_schrock()

    if Mo_N2_energy == 9999:
        return 9999, None, None, None

    # Handle the error and return if xtb did not converge
    if None in (Mo_N2_energy, Mo_NH3_energy):
        raise Exception(f"None of the XTB calculations converged")
    De = (((Mo_N2_energy + NH3_ENERGY) - (Mo_NH3_energy + N2_ENERGY))) * hartree2kcalmol
    print(f"diff energy: {De}")

    return De, Mo_N2_3d_geom, Mo_NH3_3d_geom, minidx


def rdkit_embed_scoring_NH3plustoNH3(ligand, scoring_args):
    """
    Score the NH3+ -> NH3 charge transfer

        Args:
            ligand (Chem.rdchem.Mol): ligand to put on Mo core
            scoring_args (dict): dict with all relevant args for xtb and general scoring

        Returns:
            De (float): Scoring energy
            Mo_NH3_geom: Geometry of intermediate
            Mo_NH3plus_3d_geom: Geometry of intermediate
            minidx (int): Index of the lowest energy conformer
    """

    idx = ligand.idx

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

    # Get mol object of Mo core + connected ligand
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_NH3 = connect_ligand(core_NH3[0], ligand_cut, NH3_flag=True)
    Mo_NH3 = Chem.AddHs(Mo_NH3)

    # Embed mol object
    Mo_NH3_3d = embed_rdkit(
        mol=Mo_NH3,
        core=core_NH3[0],
        numConfs=scoring_args["n_confs"],
    )

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3+"
        scoring_args["charge"] = smi_dict["Mo_NH3+"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3+"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        Mo_NH3plus_energy, Mo_NH3plus_3d_geom, minidx = optimizer.optimize_schrock()

    # THIS VALUE IS HARDCODED IN xtb_pre_optimize!
    if Mo_NH3plus_energy == 9999:
        return 9999, None, None, None

    # Remove higher energy conformers from mol object
    discard_conf = [x for x in range(len(Mo_NH3_3d.GetConformers())) if x != minidx]
    for elem in discard_conf:
        Mo_NH3_3d.RemoveConformer(elem)

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        Mo_NH3_energy, Mo_NH3_3d_geom, minidx = optimizer.optimize_schrock()

    if Mo_NH3_energy == 9999:
        return 9999, None, None, None

    # Handle the error and return if xtb did not converge
    if None in (Mo_NH3_energy, Mo_NH3plus_energy):
        raise Exception(f"None of the XTB calculations converged")
    De = (Mo_NH3_energy - Mo_NH3plus_energy + CP_RED_ENERGY) * hartree2kcalmol
    print(f"diff energy: {De}")

    return De, Mo_NH3_3d_geom, Mo_NH3plus_3d_geom, minidx


if __name__ == "__main__":

    # Current dir:
    gen = Path("/home/magstr/generation_data/prod_new15_large_0/GA50.pkl")

    # 370399_submitted.pkl
    with open(gen, "rb") as f:
        gen0 = pickle.load(f)
    ind = gen0.new_molecules[10]

    dic = {
        "cpus_per_task": 3,
        "n_confs": 1,
        "cleanup": False,
        "output_dir": "debug",
        "method": "2",
    }

    # METHYL SIMPLE
    ind = Individual(rdkit_mol=Chem.MolFromSmiles("CN"), cut_idx=1, idx=(0, 0))

    # The three scoring functions
    # res = rdkit_embed_scoring_NH3toN2(ind, dic)
    # res = rdkit_embed_scoring_NH3plustoNH3(ind, dic)
    # res = rdkit_embed_scoring(ind, dic)
    print("YOU MAAAADE IT")
