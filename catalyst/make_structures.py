from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from io import StringIO

import sys
import os


from rdkit import RDLogger

from my_utils import my_utils

RDLogger.DisableLog("rdApp.*")

# Julius files for testing
# ts_file = os.path.join(".", "input_files/ts7_dummy.sdf")
# ts_dummy = Chem.SDMolSupplier("input_files/ts7_dummy.sdf", removeHs=False, sanitize=True)[0]


def connect_cat_2d(mol_with_dummy, cat):
    """Replaces Dummy Atom [*] in Mol with Cat via tertiary Amine, return list of all possible regioisomers"""
    dummy = Chem.MolFromSmiles("*")
    mols = []
    cat = Chem.AddHs(cat)
    AllChem.AssignStereochemistry(cat)
    tert_amines = cat.GetSubstructMatches(Chem.MolFromSmarts("[#7X3;H0;D3;!+1]"))
    if len(tert_amines) == 0:
        raise Exception(
            f"{Chem.MolToSmiles(Chem.RemoveHs(cat))} constains no tertiary amine."
        )
    for amine in tert_amines:
        mol = AllChem.ReplaceSubstructs(
            mol_with_dummy, dummy, cat, replacementConnectionPoint=amine[0]
        )[0]
        quart_amine = mol.GetSubstructMatch(Chem.MolFromSmarts("[#7X4;H0;D4;!+1]"))[0]
        mol.GetAtomWithIdx(quart_amine).SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        mol.RemoveAllConformers()
        mols.append(mol)
    return mols


def frags2bonded(mol, atoms2join=((1, 11), (11, 29))):
    make_bonded = Chem.EditableMol(mol)
    for atoms in atoms2join:
        i, j = atoms
        make_bonded.AddBond(i, j)
    mol_bonded = make_bonded.GetMol()
    # Chem.SanitizeMol(mol_bonded)
    return mol_bonded


def bonded2frags(mol, atoms2frag=((1, 11), (11, 29))):
    make_frags = Chem.EditableMol(mol)
    for atoms in atoms2frag:
        i, j = atoms
        make_frags.RemoveBond(i, j)
    mol_frags = make_frags.GetMol()
    # Chem.SanitizeMol(mol_frags)
    return mol_frags


def ConstrainedEmbedMultipleConfsMultipleFrags(
    mol,
    core,
    numConfs=10,
    useTethers=True,
    coreConfId=-1,
    randomseed=2342,
    getForceField=AllChem.UFFGetMoleculeForceField,
    numThreads=1,
    force_constant=1e3,
    pruneRmsThresh=1,
    atoms2join=((1, 11), (11, 29)),
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
        print(coordMap, Chem.MolToSmiles(mol_bonded))
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


if __name__ == "__main__":

    # Driver code for testing and debugging constrained embedd.

    file_name = "../data/ZINC_first_1000.smi"
    with open(file_name, "r") as file:
        data = file.readlines()
        cat = Chem.MolFromSmiles(data[2])

    # catalysts = connect_cat_2d(ts_dummy, cat)

    # My own struct:
    file = "../templates/core_dummy.sdf"
    core = Chem.SDMolSupplier(file, removeHs=False, sanitize=False)
    catalysts = my_utils.connect_ligand(core[0], cat)

    if len(catalysts) > 1:
        print(
            f"{Chem.MolToSmiles(Chem.RemoveHs(cat))} contains more than one possible ligand"
        )
    catalyst = catalysts[0]

    # Embed TS
    ts3d = ConstrainedEmbedMultipleConfsMultipleFrags(
        mol=catalyst,
        core=core[0],
        numConfs=2,
        pruneRmsThresh=0.1,
        force_constant=1e12,
    )

    with open("test_embedd.mol", "w+") as f:
        f.write(Chem.MolToMolBlock(ts3d))

    print("Done with example")
