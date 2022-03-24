import random
import sys
import os


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from io import StringIO

from rdkit import RDLogger

from my_utils import my_utils

RDLogger.DisableLog("rdApp.*")


def connect_ligand(core, ligand, NH3_flag=False):
    """
    Function that takes two mol objects at creates a core with ligand.
    Args:
        core (mol): The core to put ligand on
        ligand (mol): The ligand to put on core

    Returns:
        mols List(mol): List of mol objects with the connect ligands.
        The length is more then one if there are multiple suitable ligands.
    """

    dummy = Chem.MolFromSmiles("*")

    # Now we need to get the indice of the atom the dummy is bound to and
    # remove the dummy atom while keeping the idx of the atom it was bound to

    # Get the dummy atom and then its atom object
    dummy_idx = ligand.GetSubstructMatch(Chem.MolFromSmiles("*"))
    atom = ligand.GetAtomWithIdx(dummy_idx[0])

    # Get the  the dummy atom has.
    neighbor_pairs = [(dummy_idx[0], x.GetIdx()) for x in atom.GetNeighbors()]

    # Get the bond between dummy and beighbor and fragment on bond
    new_bond = ligand.GetBondBetweenAtoms(
        neighbor_pairs[0][0], neighbor_pairs[0][1]
    ).GetIdx()
    frag = Chem.FragmentOnBonds(ligand, [new_bond], addDummies=False)
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # Put the ligand won the core with specified bonding atom in the ligand.
    mol = AllChem.ReplaceSubstructs(
        core,
        dummy,
        frags[0],
        replaceAll=True,
        replacementConnectionPoint=neighbor_pairs[0][1],
    )[0]

    if NH3_flag:
        mol.GetAtomWithIdx(23).SetFormalCharge(1)

    # Sanitation ensures that it is a reasonable molecule.
    Chem.SanitizeMol(mol)

    # If this is not done, the ligand i placed in zero.
    # This command removes the 3d coordinates of the core such that it is displayed well
    mol.RemoveAllConformers()

    # Show final result for debug
    # img = Draw.MolsToImage(mols)
    # img.show()
    return mol


def create_ligands(ligand):
    """
    Takes mol object and splits into fragments that can bind to a tertiary
    amine on the Mo core.
    Args:
        ligand (mol):

    Returns:
        ligands List(mol):

    """
    # TODO AllChem.ReplaceCore() could be used here instead

    # A smile indicating the dummy atoms on the core
    dummy = Chem.MolFromSmiles("*")

    # Create explicit hydrogens and sterechemistry i dont know what does.
    ligand = Chem.AddHs(ligand)
    AllChem.AssignStereochemistry(ligand)

    # Look for teriary amines in the input ligand.
    tert_amines = ligand.GetSubstructMatches(Chem.MolFromSmarts("[#7X3;H0;D3;!+1]"))
    if len(tert_amines) == 0:
        raise Exception(
            f"{Chem.MolToSmiles(Chem.RemoveHs(ligand))} constains no tertiary amine."
        )

    # Try different amines until one works.
    for amine in tert_amines:

        # Get the neigbouring bonds to the amine
        atom = ligand.GetAtomWithIdx(amine[0])
        # Create list of tuples that contain the amine idx and idx of each of the three
        # neighbors.
        indices = [(amine[0], x.GetIdx()) for x in atom.GetNeighbors()]

        # Get the bonds to the neighbors.
        bonds = []
        for atoms in indices:
            bonds.append(ligand.GetBondBetweenAtoms(atoms[0], atoms[1]).GetIdx())

        # Get the fragments from breaking the amine bonds. If the fragments connected to the tertiary
        # amine, are connected, you only carve out the N and get three dummy locations
        frag = Chem.FragmentOnBonds(
            ligand, bonds, addDummies=True, dummyLabels=[(1, 1), (1, 1), (1, 1)]
        )
        frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
        # Handle multiple dummies.

        # Check for frags with multiple dummy atoms.
        smart = "[1*]"
        # Initialize pattern
        patt = Chem.MolFromSmarts(smart)

        # Get list of ligands with only one dummy atom.
        # this also excludes the remaining tertiary amin
        ligands = [
            struct for struct in frags if len(struct.GetSubstructMatches(patt)) == 1
        ]

        NH2_mol = Chem.MolFromSmiles("[NH2]")

        ligand = random.choice(ligands)
        ligand = AllChem.ReplaceSubstructs(
            ligand, dummy, NH2_mol, replacementConnectionPoint=0
        )[0]

        lig = Chem.MolFromSmiles(Chem.MolToSmiles(ligand))

        # If valid ligands were found, break for loop and return ligands
        if ligands:
            break

    save_ligand_smiles = False
    if save_ligand_smiles:
        with open("test_ligand.smi", "w") as f:
            f.write(Chem.MolToSmiles(Chem.RemoveHs(ligands[0])))

    return lig


def create_primaryamine_ligand(ligand):
    """
    Takes mol object and splits it based on a primary amine such that the frags can connect to
    the primary amine on the Mo core.
    Args:
        ligand (mol):

    Returns:
        cut_ligand mol:

    """
    # TODO AllChem.ReplaceCore() could be used here instead

    # A smile indicating the dummy atoms on the core
    dummy = Chem.MolFromSmiles("*")

    # Create explicit hydrogens and sterechemistry i dont know what does.
    ligand = Chem.AddHs(ligand)
    AllChem.AssignStereochemistry(ligand)

    # Look for primary amines in the input ligand.
    prim_amines = ligand.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2;D3;!+1]"))
    if len(prim_amines) == 0:
        raise Exception(
            f"{Chem.MolToSmiles(Chem.RemoveHs(ligand))} constains no Primary amine."
        )

    # Get the neigbouring bonds to the amine
    atom = ligand.GetAtomWithIdx(prim_amines[0][0])
    # Create list of tuples that contain the amine idx and idx of each of the three
    # neighbors.
    indices = [
        (prim_amines[0][0], x.GetIdx())
        for x in atom.GetNeighbors()
        if x.GetAtomicNum() != 1
    ][0]

    # Get the bonds to the neighbors.
    bond = []
    bond.append(ligand.GetBondBetweenAtoms(indices[0], indices[1]).GetIdx())

    # Get the two fragments, the ligand and the NH2
    frag = Chem.FragmentOnBonds(ligand, bond, addDummies=True, dummyLabels=[(1, 1)])
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # Find frag that is NH2+dummy
    smart = "[1*][N]([H])([H])"
    # Initialize pattern
    patt = Chem.MolFromSmarts(smart)

    # Get the ligand that is not NH2
    ligands = [struct for struct in frags if len(struct.GetSubstructMatches(patt)) == 0]

    return ligands


def create_prim_amine(input_ligand):
    '''
    A function that takes a ligand and splits one a nitrogen bond, and then gives
    a ligand out that has an NH2 and a cut_idx that specifies where to cut before
    putting on the Mo core
    Args:
        input_ligand (mol):

    Returns:
        output_ligand (mol):
        prim_amine_index[0] tuple(int):
    '''
    dummy = Chem.MolFromSmiles("*")

    input_ligand = Chem.AddHs(input_ligand)
    matches = input_ligand.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2,H1,H0;!+1]"))

    # Randomly select one of the amines.
    match = random.choice(matches)

    # Get the neigbouring bonds to the amine
    atom = input_ligand.GetAtomWithIdx(match[0])
    # Create list of tuples that contain the amine idx and idx of each of the three
    # neighbors.
    indices = [
        (match[0], x.GetIdx()) for x in atom.GetNeighbors() if x.GetAtomicNum() != 1
    ]

    bond = []
    # Randomly select one of the bonds that are not to hydrogen
    atoms = random.choice(indices)
    bond.append(input_ligand.GetBondBetweenAtoms(*atoms).GetIdx())

    # Get the fragments from breaking the amine bonds. If the fragments connected to the tertiary
    # amine, are connected, you only carve out the N and get three dummy locations
    frag = Chem.FragmentOnBonds(
        input_ligand, bond, addDummies=True, dummyLabels=[(1, 1)]
    )
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    if len(frags) == 1:
        ligand = [frags[0]]
    else:
        ligand = [
            struct
            for struct in frags
            if len(struct.GetSubstructMatches(Chem.MolFromSmarts("[1*][N]"))) == 0
        ]

    if not ligand:
        raise Exception(f"Primary amine splitting went wrong.")

    NH2_mol = Chem.MolFromSmiles("[NH2]")
    lig = AllChem.ReplaceSubstructs(
        ligand[0], dummy, NH2_mol, replacementConnectionPoint=0, replaceAll=True
    )[0]
    output_ligand = Chem.MolFromSmiles(Chem.MolToSmiles(lig))

    smi = Chem.MolToSmiles(output_ligand)
    # Get idx where to cut
    prim_amine_index = output_ligand.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2]"))
    if len(prim_amine_index) > 1:
        print(
            f"There are several primary amines to cut at with idxs: {prim_amine_index}"
        )

    return output_ligand, prim_amine_index[0]


def create_dummy_ligand(ligand, cut_idx=None):
    """
    Takes mol object and splits it based on an amine such that the frags can connect to
    the tertiary amine on the Mo core.
    Args:
        cut_idx:
        ligand (mol):
    Returns:
        ligands List(mol) :
    """
    # TODO AllChem.ReplaceCore() could be used here instead

    # A smile indicating the dummy atoms on the core
    dummy = Chem.MolFromSmiles("*")

    # Create explicit hydrogens and sterechemistry i dont know what does.
    ligand = Chem.AddHs(ligand)
    AllChem.AssignStereochemistry(ligand)

    # Look for amines to split on
    amines = ligand.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2;!+1]"))
    if len(amines) == 0:
        raise Exception(
            f"{Chem.MolToSmiles(Chem.RemoveHs(ligand))} constains no primary amine"
        )

    # Get the neigbouring bonds to the amine
    atom = ligand.GetAtomWithIdx(cut_idx[0])
    # Create list of tuples that contain the amine idx and idx of each of the three
    # neighbors.
    indices = [
        (amines[0][0], x.GetIdx()) for x in atom.GetNeighbors() if x.GetAtomicNum() != 1
    ][0]

    # Get the bonds to the neighbors.
    bond = []
    bond.append(ligand.GetBondBetweenAtoms(indices[0], indices[1]).GetIdx())

    # Get the two fragments, the ligand and the NH2
    frag = Chem.FragmentOnBonds(ligand, bond, addDummies=True, dummyLabels=[(1, 1)])
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # Find frag that is NH2+dummy
    smart = "[1*][N]([H])([H])"
    # Initialize pattern
    patt = Chem.MolFromSmarts(smart)

    # Get the ligand that is not NH2
    ligands = [struct for struct in frags if len(struct.GetSubstructMatches(patt)) == 0]

    return ligands[0]


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

    # Take the input ligand and split it based on a primary amine
    ligands = create_ligands(cat)
    catalysts = connect_ligand(core[0], ligands)

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
