# -*- coding: utf-8 -*-
"""SubmitIt module handling the scoring of molecule candidates."""
import copy
import json
import logging
import os
import socket
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDistGeom

from utils.utils import energy_filter
from utils.xtb import xtb_calculate

_logger = logging.getLogger(__name__)

scoring_dir = os.path.dirname(__file__)
sys.path.append(scoring_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math

from utils.make_structures import (
    connect_ligand,
    create_dummy_ligand,
    embed_rdkit,
    remove_N2,
    remove_NH3,
)

PRE_OPT = {
    "gfn": 2,
    "opt": True,
}

HEADER1 = ["Conf-ID", "GFN-2 OPT [Hartree]"]
ROW_FORMAT1 = "{:>15}{:>25}"

from utils.utils import cd
from utils.xtb_utils import XTB_optimize_schrock

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
hartree2kcalmol = 627.51

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))

with open(str(source / "intermediate_smiles.json"), "r", encoding="utf-8") as f:
    smi_dict = json.load(f)
"""dict:
Dictionary that contains the smiles string for each N-related intermediate
and the charge and spin for the specific intermediate
"""


def rdkit_embed_scoring(ligand, scoring_args):
    """Score the NH3 -> N2 binding.

    Args:
        ligand (Chem.rdchem.Mol): ligand to put on Mo core
        scoring_args (dict): dict with all relevant args for xtb and general scoring

    Returns:
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
        mol=Mo_N2_NH3,
        core=core_N2_NH3[0],
        numConfs=scoring_args["n_confs"],
        pruneRmsThresh=scoring_args["RMS_thresh"],
    )

    # Go into the output directory
    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_N2_NH3"
        scoring_args["charge"] = smi_dict["Mo_N2_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_N2_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_N2_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol1, energies = optimizer.optimize_schrock()

    # Get conformers
    confs = optimized_mol1.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    # Remove conformers that are too far from the minimum energy conf
    energies, new_mol = energy_filter(confs, energies, optimized_mol1, scoring_args)

    # Create duplicate mol object for removing conformers
    single_conf = copy.deepcopy(new_mol)

    # During GA scoring, only one conformer is needed for the next step.
    # The rest of the conformers are discarded.
    if scoring_args.get("ga_scoring", False):
        # Remove higher energy conformes from mol object
        minidx = np.argmin(energies)
        discard_conf = [
            x for x in range(len(single_conf.GetConformers())) if x != minidx
        ]
        for elem in discard_conf:
            single_conf.RemoveConformer(elem)

    # Remove N2 on the full embedding
    Mo_NH3_3d = remove_N2(single_conf)
    Mo_NH3_3d = Chem.AddHs(Mo_NH3_3d)

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol2, energies2 = optimizer.optimize_schrock()

    confs = optimized_mol2.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    energies2, new_mol2 = energy_filter(confs, energies2, optimized_mol2, scoring_args)

    energy_diff = (energies.min() - (energies2.min() + N2_ENERGY)) * hartree2kcalmol
    print(f"Score for top scoring conformer: {energy_diff}")

    en_dict = {"energy1": energies, "energy2": energies2, "score": energy_diff}

    return new_mol, new_mol2, en_dict


def rdkit_embed_scoring_NH3toN2(ligand, scoring_args):
    """Score the NH3 -> N2 exchange.

    Args:
        ligand (Chem.rdchem.Mol): ligand to put on Mo core
        scoring_args (dict): dict with all relevant args for xtb and general scoring

    Returns:
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
        pruneRmsThresh=scoring_args["RMS_thresh"],
    )

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol1, energies = optimizer.optimize_schrock()

    confs = optimized_mol1.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    # Remove conformers that are too far from the minimum energy conf
    energies, new_mol = energy_filter(confs, energies, optimized_mol1, scoring_args)

    single_conf = copy.deepcopy(new_mol)

    minidx = np.argmin(energies)
    discard_conf = [x for x in range(len(single_conf.GetConformers())) if x != minidx]
    for elem in discard_conf:
        single_conf.RemoveConformer(elem)

    # Replace NH3 with N2
    Mo_N2 = Chem.ReplaceSubstructs(
        single_conf,
        Chem.AddHs(Chem.MolFromSmarts("[NH3]")),
        Chem.MolFromSmarts("N#N"),
        replaceAll=True,
    )[0]

    # Get bare Mo to use as embed reference
    Mo_3d = remove_NH3(single_conf)
    Mo_3d = Chem.AddHs(Mo_3d)

    # Change charge of the N bound to Mo to ensure sanitation works
    match = Mo_N2.GetSubstructMatch(Chem.MolFromSmarts("[Mo]N#N"))
    Mo_N2.GetAtomWithIdx(match[1]).SetFormalCharge(1)

    # Embed catalyst
    if scoring_args.get("ga_scoring", False):
        Mo_N2_3d = embed_rdkit(
            mol=Mo_N2, core=Mo_3d, numConfs=1, pruneRmsThresh=scoring_args["RMS_thresh"]
        )
    else:
        Mo_N2_3d = embed_rdkit(
            mol=Mo_N2,
            core=core[0],
            numConfs=scoring_args["n_confs"],
            pruneRmsThresh=scoring_args["RMS_thresh"],
        )

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_N2"
        scoring_args["charge"] = smi_dict["Mo_N2"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_N2"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_N2_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol2, energies2 = optimizer.optimize_schrock()

    confs = optimized_mol2.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    energies2, new_mol2 = energy_filter(confs, energies2, optimized_mol2, scoring_args)

    energy_diff = (
        ((energies2.min() + NH3_ENERGY) - (energies.min() + N2_ENERGY))
    ) * hartree2kcalmol
    print(f"Score for top scoring conformer: {energy_diff}")

    en_dict = {"energy1": energies, "energy2": energies2, "score": energy_diff}

    return new_mol, new_mol2, en_dict


def rdkit_embed_scoring_NH3plustoNH3(ligand, scoring_args):
    """Score the NH3+ -> NH3 charge transfer.

    Args:
        ligand (Chem.rdchem.Mol): ligand to put on Mo core
        scoring_args (dict): dict with all relevant args for xtb and general scoring
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
        pruneRmsThresh=scoring_args["RMS_thresh"],
    )

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3+"
        scoring_args["charge"] = smi_dict["Mo_NH3+"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3+"]["spin"]
        optimizer = XTB_optimize_schrock(mol=Mo_NH3_3d, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol1, energies = optimizer.optimize_schrock()

    confs = optimized_mol1.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    # Remove fonformers that are too far from the minimum energy conf
    energies, new_mol = energy_filter(confs, energies, optimized_mol1, scoring_args)

    single_conf = copy.deepcopy(new_mol)
    if scoring_args.get("ga_scoring", False):
        # Copy into new mol object
        # Remove higher energy conformes from mol object
        minidx = np.argmin(energies)
        discard_conf = [
            x for x in range(len(single_conf.GetConformers())) if x != minidx
        ]
        for elem in discard_conf:
            single_conf.RemoveConformer(elem)

    with cd(scoring_args["output_dir"]):

        # Instantiate optimizer class
        scoring_args["name"] = f"{idx[0]:03d}_{idx[1]:03d}_Mo_NH3"
        scoring_args["charge"] = smi_dict["Mo_NH3"]["charge"]
        scoring_args["uhf"] = smi_dict["Mo_NH3"]["spin"]
        optimizer = XTB_optimize_schrock(mol=single_conf, scoring_options=scoring_args)

        # Perform calculation
        optimized_mol2, energies2 = optimizer.optimize_schrock()

    confs = optimized_mol2.GetConformers()
    if len(confs) == 0:
        return None, None, {"energy1": None, "energy2": None, "score": np.nan}

    energies2, new_mol2 = energy_filter(confs, energies2, optimized_mol2, scoring_args)

    energy_diff = (energies2.min() - energies.min() + CP_RED_ENERGY) * hartree2kcalmol

    print(f"Score for top scoring conformer: {energy_diff}")

    en_dict = {"energy1": energies, "energy2": energies2, "score": energy_diff}

    return new_mol, new_mol2, en_dict


def calculate_score_logP(ind):
    start = time.time()

    mol = Chem.AddHs(ind.rdkit_mol)
    try:
        cid = rdDistGeom.EmbedMolecule(mol, useRandomCoords=True)
        if cid != 0:
            raise Exception("Embedding failed")
        logP = Descriptors.MolLogP(mol)
    except Exception as e:
        error = str(e)
        logP = math.nan
    ind.score = logP
    return ind


def calculate_score(ind, n_cores: int = 1, envvar_scratch: str = "SCRATCH"):

    print(socket.gethostname())

    # Setup scrach directory
    scratch = os.environ.get(envvar_scratch, ".")
    calc_dir = Path(scratch)
    start_time = time.time()
    jobid = os.getenv("SLURM_ARRAY_ID", str(uuid.uuid4()))
    calc_dir = calc_dir / jobid
    calc_dir.mkdir(exist_ok=True)

    # Setup logging to stdout
    _logger.setLevel(logging.INFO)
    _logger.addHandler(logging.StreamHandler())
    _logger.warning(socket.gethostname())
    _logger.info(f"Calculating score for  SMILES: {ind.smiles}\n")

    start = time.time()

    ind.rdkit_mol = Chem.AddHs(ind.rdkit_mol)
    cid = rdDistGeom.EmbedMultipleConfs(
        ind.rdkit_mol,
        numConfs=1,
        useRandomCoords=True,
        pruneRmsThresh=0.25,
    )

    _logger.info(f"Embedded {ind.rdkit_mol.GetNumConformers()} conformers.")

    # Determine connectivity for mol4
    mol_adj = Chem.GetAdjacencyMatrix(ind.rdkit_mol)

    mol_atoms = [a.GetSymbol() for a in ind.rdkit_mol.GetAtoms()]

    _logger.info(f"{ROW_FORMAT1.format(*HEADER1)}")
    for conf in ind.rdkit_mol.GetConformers():
        cid = conf.GetId()
        mol_coords = conf.GetPositions()

        # Pre-optimization of mol
        _, mol_opt_coords, energy = xtb_calculate(
            atoms=mol_atoms,
            coords=mol_coords,
            options=PRE_OPT,
            scr=calc_dir,
            n_cores=4,
        )

    ind.score = energy

    return ind
