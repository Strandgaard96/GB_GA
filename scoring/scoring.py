from rdkit import Chem

import os
import sys
import json
import pickle

catalyst_dir = os.path.dirname(__file__)
sys.path.append(catalyst_dir)

# from .xtb_utils import xtb_optimize
from my_utils.my_xtb_utils import xtb_optimize
from my_utils.my_utils import cd
from .make_structures import (
    connect_ligand,
    create_ligands,
    create_primaryamine_ligand,
    embed_rdkit,
    create_dummy_ligand,
)

hartree2kcalmol = 627.5094740631
CORE_ELECTRONIC_ENERGY = -32.698
NH3_ENERGY = -4.4120

# My own structs:
file = "templates/core_dummy.sdf"
core = Chem.SDMolSupplier(file, removeHs=False, sanitize=False)

file_NH3 = "templates/core_NH3_dummy.sdf"
core_NH3 = Chem.SDMolSupplier(file_NH3, removeHs=False, sanitize=False)

with open("data/intermediate_smiles.json", "r", encoding="utf-8") as f:
    smi_dict = json.load(f)


def rdkit_embed_scoring(
    ligand, idx=(0, 0), ncpus=1, n_confs=10, cleanup=False, output_dir="."
):
    """My driver scoring function."""

    # Create ligand based on a primary amine
    #ligand_cut = create_primaryamine_ligand(ligand.rdkit_mol)[0]
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    catalyst = connect_ligand(core[0], ligand_cut)

    # Embed catalyst
    catalyst_3d = embed_rdkit(
        mol=catalyst,
        core=core[0],
        numConfs=n_confs,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    print("Done with embedding of catalyst")

    with cd(output_dir):
        catalyst_3d_energy, catalyst_3d_geom = xtb_optimize(
            catalyst_3d,
            gbsa="benzene",
            charge=smi_dict["Mo"]["charge"],
            spin=smi_dict["Mo"]["spin"],
            opt_level="loose",
            name=f"{idx[0]:03d}_{idx[1]:03d}_catalyst",
            input=os.path.join("../../..", "templates/input_files/constr.inp"),
            numThreads=ncpus,
            cleanup=cleanup,
        )
        print("catalyst energy:", catalyst_3d_energy)

    print("Start scoring of Mo_NH3 intermediate")

    catalyst_NH3 = connect_ligand(core_NH3[0], ligand_cut, NH3_flag=True)

    # Embed catalyst
    Mo_NH3_3d = embed_rdkit(
        mol=catalyst_NH3,
        core=core_NH3[0],
        numConfs=n_confs,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    print("Done with embedding of Mo_NH3")

    with cd(output_dir):
        Mo_NH3_3d_energy, Mo_NH3_3d_geom = xtb_optimize(
            Mo_NH3_3d,
            gbsa="benzene",
            charge=smi_dict["Mo_NH3"]["charge"],
            spin=smi_dict["Mo_NH3"]["spin"],
            opt_level="loose",
            name=f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3",
            input=os.path.join("../../..", "templates/input_files/constr.inp"),
            numThreads=ncpus,
            cleanup=cleanup,
        )
        print("Mo_NH3 energy:", Mo_NH3_3d_energy)

    # Scoring function based on the energies
    # print(f"All energies: cat: {catalyst_3d_energy} ligand: {ligand_3d_energy}")
    # ligand_3d_energy = 0

    # De = (catalyst_3d_energy - CORE_ELECTRONIC_ENERGY - ligand_3d_energy) * hartree2kcalmol
    De_new = ((catalyst_3d_energy + NH3_ENERGY) - Mo_NH3_3d_energy) * hartree2kcalmol
    return De_new, (catalyst_3d_geom, catalyst_3d_energy)


def runner_for_test():
    # runner for putting proposed ligand on core.
    file = "../templates/core_dummy.sdf"
    core = Chem.SDMolSupplier(file, removeHs=False, sanitize=False)

    file_name = "../data/ZINC_first_1000.smi"
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            mol_list.append(Chem.MolFromSmiles(smiles))
    # mols = connect_ligand(core[0], mol_list[50])

    ligands = create_ligands(mol_list[50])

    # Put ligands on the core and do xtb.

    # Get idx of atom that should attach

    # Get path to ligand
    # ligand =

    # molSimplify_scoring(ligand, idx=(0, 0), ncpus=1, cleanup=False, output_dir=".")


if __name__ == "__main__":
    # runner_for_test()
    file_name = "../data/ZINC_first_1000.smi"
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            mol_list.append(Chem.MolFromSmiles(smiles))

    lig = create_ligands(mol_list[1])

    # Useful for debugging failed scoring. Load the pickle file
    # From the failed calc.
    # with open(
    #    "/home/magstr/generation_prim_amine/scoring_tmp/4772867_8_submitted.pkl", "rb"
    # ) as handle:
    #    b = pickle.load(handle)

    rdkit_embed_scoring(lig)
