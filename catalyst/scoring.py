from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import os
import copy
import numpy as np
import sys
import json
from io import StringIO
import pickle

catalyst_dir = os.path.dirname(__file__)
sys.path.append(catalyst_dir)

# from .xtb_utils import xtb_optimize
from my_utils.my_xtb_utils import xtb_optimize
from my_utils.my_utils import cd
from .make_structures import (
    connect_cat_2d,
    ConstrainedEmbedMultipleConfsMultipleFrags,
    connect_ligand,
    create_ligands,
)
from .make_structures import create_primaryamine_ligand

frag_energies = np.sum(
    [-8.232710038092, -19.734652802142, -32.543971411432]
)  # 34 atoms
hartree2kcalmol = 627.5094740631

CORE_ELECTRONIC_ENERGY = -32.698
NH3_ENERGY = -4.4120

ts_file = os.path.join(catalyst_dir, "input_files/ts7_dummy.sdf")
int_file = os.path.join(catalyst_dir, "input_files/int5_dummy.sdf")

ts_dummy = Chem.SDMolSupplier(ts_file, removeHs=False, sanitize=True)[0]

# My own structs:
file = "templates/core_dummy.sdf"
core = Chem.SDMolSupplier(file, removeHs=False, sanitize=False)

file_NH3 = "templates/core_NH3_dummy.sdf"
core_NH3 = Chem.SDMolSupplier(file_NH3, removeHs=False, sanitize=False)

with open("data/intermediate_smiles.json", "r", encoding="utf-8") as f:
    smi_dict = json.load(f)


def ts_scoring(cat, idx=(0, 0), ncpus=1, n_confs=10, cleanup=False, output_dir="."):
    """Calculates electronic energy difference in kcal/mol between TS and reactants

    Args:
        cat (rdkit.Mol): Molecule containing one tertiary amine
        n_confs (int, optional): Nubmer of confomers used for embedding. Defaults to 10.
        cleanup (bool, optional): Clean up files after calculation.
                                  Defaults to False, needs to be False to work with submitit.

    Returns:
        Tuple: Contains energy difference, Geom of TS and Geom of Cat
    """

    ts2ds = connect_cat_2d(ts_dummy, cat.rdkit_mol)
    if len(ts2ds) > 1:
        print(
            f"{Chem.MolToSmiles(Chem.RemoveHs(cat.rdkit_mol))} contains more than one tertiary amine"
        )
    ts2d = ts2ds[0]

    # Embed TS
    ts3d = ConstrainedEmbedMultipleConfsMultipleFrags(
        mol=ts2d,
        core=ts_dummy,
        numConfs=n_confs,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    # logger.debug('Running xtb with catalyst_dir %s',catalyst_dir)
    # Calc Energy of TS
    with cd(output_dir):
        ts3d_energy, ts3d_geom = xtb_optimize(
            ts3d,
            gbsa="methanol",
            opt_level="loose",
            name=f"{idx[0]:03d}_{idx[1]:03d}_ts",
            input=os.path.join(catalyst_dir, "input_files/constr.inp"),
            numThreads=ncpus,
            cleanup=cleanup,
        )

    # Embed Catalyst
    cat3d = copy.deepcopy(cat.rdkit_mol)
    cat3d = Chem.AddHs(cat3d)
    cids = Chem.rdDistGeom.EmbedMultipleConfs(
        cat3d, numConfs=n_confs, pruneRmsThresh=0.1
    )
    if len(cids) == 0:
        raise ValueError(
            f"Could not embed catalyst {Chem.MolToSmiles(Chem.RemoveHs(cat))}"
        )

    # Calc Energy of Cat
    with cd(output_dir):
        cat3d_energy, cat3d_geom = xtb_optimize(
            cat3d,
            gbsa="methanol",
            opt_level="loose",
            name=f"{idx[0]:03d}_{idx[1]:03d}_cat",
            numThreads=ncpus,
            cleanup=cleanup,
        )

    # Calculate electronic activation energy
    print(ts3d_energy, frag_energies, cat3d_energy, hartree2kcalmol)
    De = (ts3d_energy - frag_energies - cat3d_energy) * hartree2kcalmol
    return De, (ts3d_geom, cat3d_geom)


def embed_rdkit(
    mol,
    core,
    numConfs=10,
    coreConfId=-1,
    randomseed=2342,
    getForceField=AllChem.UFFGetMoleculeForceField,
    numThreads=1,
    force_constant=1e3,
    pruneRmsThresh=1,
):
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    sio = sys.stderr = StringIO()
    # if not AllChem.UFFHasAllMoleculeParams(mol):
    #    raise Exception(Chem.MolToSmiles(mol), sio.getvalue())

    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    cids = AllChem.EmbedMultipleConfs(
        mol=mol,
        numConfs=numConfs,
        coordMap=coordMap,
        randomSeed=randomseed,
        numThreads=numThreads,
        pruneRmsThresh=pruneRmsThresh,
        useRandomCoords=False,
    )
    Chem.SanitizeMol(mol)

    cids = list(cids)
    if len(cids) == 0:
        print(coordMap, Chem.MolToSmiles(mol))
        raise ValueError("Could not embed molecule.")

    algMap = [(j, i) for i, j in enumerate(match)]

    # rotate the embedded conformation onto the core:
    for cid in cids:
        rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=cid, ignoreInterfragInteractions=False
        )
        for i, _ in enumerate(match):
            ff.UFFAddPositionConstraint(i, 0, force_constant)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    return mol


def rdkit_embed_scoring(
    ligand, idx=(0, 0), ncpus=1, n_confs=10, cleanup=False, output_dir="."
):
    """My driver scoring function."""

    # get new core and pass file to xtb_optimize
    # logger.debug('Running xtb with catalyst_dir %s',catalyst_dir)

    # TEst primary amine gen
    ligand_cut = create_primaryamine_ligand(ligand.rdkit_mol)[0]
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
    with open(
        "/home/magstr/generation_prim_amine/scoring_tmp/4772867_8_submitted.pkl", "rb"
    ) as handle:
        b = pickle.load(handle)

    rdkit_embed_scoring(lig)
