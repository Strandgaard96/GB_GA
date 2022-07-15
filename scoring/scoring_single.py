# -*- coding: utf-8 -*-
""" Scoring module
Module handling the driver for scoring single ligand candidates.
Contains various global variables that should be available to the scoring
function at all times

"""
import os
import sys
import json
import pickle

from rdkit import Chem

catalyst_dir = os.path.dirname(__file__)
sys.path.append(catalyst_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_utils.my_xtb_utils_single import xtb_pre_optimize
from my_utils.my_utils import cd
from make_structures import (
    connect_ligand,
    create_ligands,
    create_prim_amine,
    create_primaryamine_ligand,
    embed_rdkit,
    create_dummy_ligand,
    connectMols,
    remove_NH3,
    remove_N2,
)
from my_utils.my_utils import Individual, Generation

hartree2kcalmol = 627.5094740631
CORE_ELECTRONIC_ENERGY = -32.698
NH3_ENERGY = -4.4260

n2_correction = 0.00
N2_ENERGY = -5.7639 - n2_correction
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


def rdkit_embed_scoring(
    ligand, idx=(0, 0), ncpus=1, n_confs=10, cleanup=False, output_dir="."
):

    # Create ligand based on a primary amine
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_N2_NH3 = connect_ligand(core_N2_NH3[0], ligand_cut, NH3_flag=True)

    # Embed catalyst
    Mo_N2_NH3_3d = embed_rdkit(
        mol=Mo_N2_NH3,
        core=core_N2_NH3[0],
        numConfs=n_confs,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    with cd(output_dir):

        # Optimization
        Mo_N2_NH3_energy, Mo_N2_NH3_3d_geom, minidx = xtb_pre_optimize(
            Mo_N2_NH3_3d,
            gbsa="benzene",
            charge=smi_dict["Mo_N2_NH3"]["charge"],
            spin=smi_dict["Mo_N2_NH3"]["spin"],
            opt_level="tight",
            name=f"{idx[0]:03d}_{idx[1]:03d}_Mo_N2_NH3",
            numThreads=ncpus,
            cleanup=cleanup,
        )
        print("catalyst energy:", Mo_N2_NH3_energy)

    # THIS VALUE IS HARDCODED IN xtb_pre_optimize!
    if Mo_N2_NH3_energy == 9999:
        return 9999, None, None

    # Now we want to remove the NH2 on the already embedded structure
    discard_conf = [x for x in range(len(Mo_N2_NH3_3d.GetConformers())) if x != minidx]
    for elem in discard_conf:
        Mo_N2_NH3_3d.RemoveConformer(elem)

    Mo_NH3_3d = remove_N2(Mo_N2_NH3_3d)
    Mo_NH3_3d = Chem.AddHs(Mo_NH3_3d)

    with cd(output_dir):
        Mo_NH3_energy, Mo_NH3_3d_geom, minidx = xtb_pre_optimize(
            Mo_NH3_3d,
            gbsa="benzene",
            charge=smi_dict["Mo_NH3"]["charge"],
            spin=smi_dict["Mo_NH3"]["spin"],
            opt_level="tight",
            name=f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3",
            numThreads=ncpus,
            cleanup=cleanup,
        )
        print("Mo energy:", Mo_NH3_energy)

    if Mo_NH3_energy == 9999:
        return 9999, None, None

    # Handle the error and return if xtb did not converge
    if None in (Mo_N2_NH3_energy, Mo_NH3_energy):
        raise Exception(f"None of the XTB calculations converged")
    De = ((Mo_N2_NH3_energy - (Mo_NH3_energy + N2_ENERGY))) * hartree2kcalmol
    print(f"diff energy: {De}")
    return De, Mo_N2_NH3_3d_geom, minidx


if __name__ == "__main__":

    # Testing HIPT ligand
    smi = "CC(C)c1cc(C(C)C)c(-c2cc(N)cc(-c3c(C(C)C)cc(C(C)C)cc3C(C)C)c2)c(C(C)C)c1"
    HIPT = Chem.AddHs(Chem.MolFromSmiles(smi))
    cut_idx = 13
    HIPT_ind = Individual(HIPT, cut_idx=cut_idx)

    rdkit_embed_scoring(HIPT_ind, n_confs=2, ncpus=16)
