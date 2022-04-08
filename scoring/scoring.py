# -*- coding: utf-8 -*-
""" Scoring module

Module handling the driver for scoring ligand candidates.
Contains various global variables that should be available to the scoring
function at all times

Todo:
    *
"""
import os
import sys
import json
import pickle

from rdkit import Chem

catalyst_dir = os.path.dirname(__file__)
sys.path.append(catalyst_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_utils.my_xtb_utils import xtb_optimize, xtb_pre_optimize
from my_utils.my_utils import cd
from make_structures import (
    connect_ligand,
    create_ligands,
    create_prim_amine,
    create_primaryamine_ligand,
    embed_rdkit,
    create_dummy_ligand,
    connectMols,
    remove_NH3
)
from my_utils.my_utils import Individual, Population

hartree2kcalmol = 627.5094740631
CORE_ELECTRONIC_ENERGY = -32.698
NH3_ENERGY = -4.4260
"""int: Module level constants
hartree2kcalmol: Handles conversion from hartree to kcal/mol
CORE_ELECTRONIC_ENERGY: The electronic energy of the Mo core with cut
ligands
NH3_ENERGY: Electronic energy of pure NH3, 
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
with open("data/intermediate_smiles.json", "r", encoding="utf-8") as f:
    smi_dict = json.load(f)
"""dict: 
Dictionary that contains the smiles string for each N-related intermediate
and the charge and spin for the specific intermediate
"""


def rdkit_embed_scoring(
    ligand, idx=(0, 0), ncpus=1, n_confs=10, cleanup=False, output_dir="."
):
    """

    Args:
        ligand:
        idx:
        ncpus:
        n_confs:
        cleanup:
        output_dir:

    Returns:

    """

    # Create ligand based on a primary amine
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    catalyst_NH3 = connect_ligand(core_NH3[0], ligand_cut, NH3_flag=True)

    # Embed catalyst
    catalyst_NH3_3d = embed_rdkit(
        mol=catalyst_NH3,
        core=core_NH3[0],
        numConfs=n_confs,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    with cd(output_dir):

        # Optimization
        catalyst_NH3_energy, catalyst_NH3_3d_geom, minidx = xtb_pre_optimize(
            catalyst_NH3_3d,
            gbsa="benzene",
            charge=smi_dict["Mo_NH3"]["charge"],
            spin=smi_dict["Mo_NH3"]["spin"],
            opt_level="tight",
            name=f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3",
            numThreads=ncpus,
            cleanup=cleanup,
        )
        print("catalyst energy:", catalyst_NH3_energy)

    # Now we want to remove the NH3 on the already embedded structure
    catalyst = remove_NH3(catalyst_NH3_3d)
    catalyst = Chem.AddHs(catalyst)

    with cd(output_dir):
        Mo_3d_energy, Mo_3d_geom, minidx = xtb_pre_optimize(
            catalyst,
            gbsa="benzene",
            charge=smi_dict["Mo"]["charge"],
            spin=smi_dict["Mo"]["spin"],
            opt_level="tight",
            name=f"{idx[0]:03d}_{idx[1]:03d}_catalyst",
            numThreads=1,
            cleanup=cleanup,
        )
        print("Mo energy:", Mo_3d_energy)

    # Handle the error and return if xtb did not converge
    if None in (catalyst_NH3_energy, Mo_3d_energy):
        raise Exception(f"XTB calculation did not converge")
    De = ((Mo_3d_energy + NH3_ENERGY) - catalyst_NH3_energy) * hartree2kcalmol
    print(f"diff energy: {(Mo_3d_energy + NH3_ENERGY) - catalyst_NH3_energy}")
    return De, Mo_3d_geom, minidx


if __name__ == "__main__":

    # runner_for_test()
    file_name = "../data/ZINC_first_1000.smi"
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            mol_list.append(Chem.MolFromSmiles(smiles))

    lig, cut_idx = create_prim_amine(mol_list[1])

    lig = Chem.MolFromSmiles("CN")
    cut_idx = 1
    ind = Individual(lig, cut_idx=cut_idx)
    # Useful for debugging failed scoring. Load the pickle file
    # From the failed calc.
    # with open(
    #    "/home/magstr/generation_prim_amine/scoring_tmp/4772867_8_submitted.pkl", "rb"
    # ) as handle:
    #    b = pickle.load(handle)

    rdkit_embed_scoring(ind, n_confs=4, ncpus=2)
