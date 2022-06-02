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
from pathlib import Path

from rdkit import Chem

catalyst_dir = os.path.dirname(__file__)
sys.path.append(catalyst_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_utils.my_xtb_utils import xtb_pre_optimize
from my_utils.my_utils import cd
from make_structures import (
    connect_ligand,
    create_ligands,
    create_prim_amine,
    embed_rdkit,
    create_dummy_ligand,
    connectMols,
    remove_NH3,
    remove_N2,
    mol_with_atom_index,
)
from my_utils.my_utils import Individual, Population

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


def unpack_args(scoring_args):
    ncpus = scoring_args["cpus_per_task"]
    n_confs = scoring_args["n_confs"]
    cleanup = scoring_args["cleanup"]
    output_dir = scoring_args["output_dir"]
    method = scoring_args["method"]
    return ncpus, n_confs, cleanup, output_dir, method


def rdkit_embed_scoring(ligand, scoring_args):

    # Unpack variables
    idx = ligand.idx
    ncpus, n_confs, cleanup, output_dir, method = unpack_args(scoring_args)

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

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
            bare=True,
            method=method,
        )
        print("catalyst energy:", Mo_N2_NH3_energy)

    # THIS VALUE IS HARDCODED IN xtb_pre_optimize!
    if Mo_N2_NH3_energy == 9999:
        return 9999, None, None, None

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
            method=method,
        )
        print("Mo energy:", Mo_NH3_energy)

    if Mo_NH3_energy == 9999:
        return 9999, None, None, None

    # Handle the error and return if xtb did not converge
    if None in (Mo_N2_NH3_energy, Mo_NH3_energy):
        raise Exception(f"None of the XTB calculations converged")
    De = ((Mo_N2_NH3_energy - (Mo_NH3_energy + N2_ENERGY))) * hartree2kcalmol
    print(f"diff energy: {De}")
    return De, Mo_N2_NH3_3d_geom, Mo_NH3_3d_geom, minidx


def rdkit_embed_scoring_NH3toN2(ligand, scoring_args):

    idx = ligand.idx
    ncpus, n_confs, cleanup, output_dir, method = unpack_args(scoring_args)

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

    # Create ligand based on a primary amine
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_NH3 = connect_ligand(core_NH3[0], ligand_cut, NH3_flag=True)

    # Embed catalyst
    Mo_NH3_3d = embed_rdkit(
        mol=Mo_NH3,
        core=core_NH3[0],
        numConfs=n_confs,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    with cd(output_dir):

        # Optimization
        Mo_NH3_energy, Mo_NH3_3d_geom, minidx = xtb_pre_optimize(
            Mo_NH3_3d,
            gbsa="benzene",
            charge=smi_dict["Mo_NH3"]["charge"],
            spin=smi_dict["Mo_NH3"]["spin"],
            opt_level="tight",
            name=f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3",
            numThreads=ncpus,
            cleanup=cleanup,
            bare=True,
            method=method,
        )
        print("catalyst energy:", Mo_NH3_energy)

    # THIS VALUE IS HARDCODED IN xtb_pre_optimize!
    if Mo_NH3_energy == 9999:
        return 9999, None, None, None

    # Now we want to remove the NH2 on the already embedded structure
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

    # Change charge of the N bound to mo
    match = Mo_N2.GetSubstructMatch(Chem.MolFromSmarts("[Mo]N#N"))
    Mo_N2.GetAtomWithIdx(match[1]).SetFormalCharge(1)

    # Embed catalyst
    Mo_N2_3d = embed_rdkit(
        mol=Mo_N2,
        core=Mo_3d,
        numConfs=1,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    with cd(output_dir):
        Mo_N2_energy, Mo_N2_3d_geom, minidx = xtb_pre_optimize(
            Mo_N2_3d,
            gbsa="benzene",
            charge=smi_dict["Mo_N2"]["charge"],
            spin=smi_dict["Mo_N2"]["spin"],
            opt_level="tight",
            name=f"{idx[0]:03d}_{idx[1]:03d}_Mo_N2",
            numThreads=ncpus,
            cleanup=cleanup,
            method=method,
        )
        print("Mo energy:", Mo_N2_energy)

    if Mo_N2_energy == 9999:
        return 9999, None, None, None

    # Handle the error and return if xtb did not converge
    if None in (Mo_N2_energy, Mo_NH3_energy):
        raise Exception(f"None of the XTB calculations converged")
    De = (((Mo_N2_energy + NH3_ENERGY) - (Mo_NH3_energy + N2_ENERGY))) * hartree2kcalmol
    print(f"diff energy: {De}")
    return De, Mo_N2_3d_geom, Mo_NH3_3d_geom, minidx


def rdkit_embed_scoring_NH3plustoNH3(ligand, scoring_args):
    idx = ligand.idx
    ncpus, n_confs, cleanup, output_dir, method = unpack_args(scoring_args)

    # Unpack gas energies
    NH3_ENERGY, N2_ENERGY, CP_RED_ENERGY = GAS_ENERGIES[scoring_args["method"]]

    # Create ligand based on a primary amine
    ligand_cut = create_dummy_ligand(ligand.rdkit_mol, ligand.cut_idx)
    Mo_NH3 = connect_ligand(core_NH3[0], ligand_cut, NH3_flag=True)

    # Embed catalyst
    Mo_NH3_3d = embed_rdkit(
        mol=Mo_NH3,
        core=core_NH3[0],
        numConfs=n_confs,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    with cd(output_dir):

        # Optimization
        Mo_NH3plus_energy, Mo_NH3plus_3d_geom, minidx = xtb_pre_optimize(
            Mo_NH3_3d,
            gbsa="benzene",
            charge=smi_dict["Mo_NH3+"]["charge"],
            spin=smi_dict["Mo_NH3+"]["spin"],
            opt_level="tight",
            name=f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3+",
            numThreads=ncpus,
            cleanup=cleanup,
            bare=True,
            method=method,
        )
        print("catalyst energy:", Mo_NH3plus_energy)

    # THIS VALUE IS HARDCODED IN xtb_pre_optimize!
    if Mo_NH3plus_energy == 9999:
        return 9999, None, None, None

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
            method=method,
        )
        print("Mo energy:", Mo_NH3_energy)

    if Mo_NH3_energy == 9999:
        return 9999, None, None, None

    # Handle the error and return if xtb did not converge
    if None in (Mo_NH3_energy, Mo_NH3plus_energy):
        raise Exception(f"None of the XTB calculations converged")
    De = (Mo_NH3_energy - Mo_NH3plus_energy + CP_RED_ENERGY) * hartree2kcalmol
    print(f"diff energy: {De}")
    return De, Mo_NH3_3d_geom, Mo_NH3plus_3d_geom, minidx


if __name__ == "__main__":

    # runner_for_test()
    file_name = "data/ZINC_first_1000.smi"
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
    # with open("debug/32597920_46_submitted.pkl", "rb") as handle:
    #    b = pickle.load(handle)

    file_noMo = "/home/magstr/Documents/GB_GA/050_017_Mo_N2_NH3/conf003/xtbopt_noMo.xyz"
    from my_utils.xyz2mol import read_xyz_file, xyz2mol, xyz2AC

    # Testing HIPT ligand
    smi = "CC(C)c1cc(C(C)C)c(-c2cc(N)cc(-c3c(C(C)C)cc(C(C)C)cc3C(C)C)c2)c(C(C)C)c1"
    HIPT = Chem.AddHs(Chem.MolFromSmiles(smi))
    cut_idx = 1
    HIPT_ind = Individual(HIPT, cut_idx=cut_idx)

    # Current dir:
    gen = Path("debug/33422997_32_submitted.pkl")

    # 370399_submitted.pkl
    with open(gen, "rb") as f:
        gen0 = pickle.load(f)
    # ind = gen0.args[0]

    rdkit_embed_scoring_NH3toN2(ind, n_confs=1, ncpus=3)
