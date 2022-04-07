# -*- coding: utf-8 -*-
"""
Module that performs the handling mol ligands. Conversion between
x-amines to dummy atoms and subsequent placement on Mo core

Todo:
    *
"""

import random
import sys
import os
from io import StringIO

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage

IPythonConsole.drawOptions.addAtomIndices = True

RDLogger.DisableLog("rdApp.*")

from my_utils import my_utils


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    Chem.Draw.MolToImage(mol, size=(400, 400)).show()
    return mol

def remove_NH3(mol):

    # Substructure match the NH3
    NH3_match = Chem.MolFromSmarts("[NH3]")
    NH3_match = Chem.AddHs(NH3_match)
    removed_mol = Chem.DeleteSubstructs(mol, NH3_match)

    return removed_mol


def getAttachmentVector(mol):
    """Search for the position of the attachment point and extract the atom index of the attachment point and the connected atom (only single neighbour supported)
    Function from https://pschmidtke.github.io/blog/rdkit/3d-editor/2021/01/23/grafting-fragments.html
    mol: rdkit molecule with a dummy atom
    return: atom indices
    """
    rindex = -1
    rindexNeighbor = -1
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            rindex = atom.GetIdx()
            neighbours = atom.GetNeighbors()
            if len(neighbours) == 1:
                rindexNeighbor = neighbours[0].GetIdx()
            else:
                print("two attachment points not supported yet")
                return None

    return rindex, rindexNeighbor


def replaceAtom(mol, indexAtom, indexNeighbor, atom_type="Br"):
    """Replace an atom with another type"""

    emol = Chem.EditableMol(mol)
    emol.ReplaceAtom(indexAtom, Chem.Atom(atom_type))
    emol.RemoveBond(indexAtom, indexNeighbor)
    emol.AddBond(indexAtom, indexNeighbor, order=Chem.rdchem.BondType.SINGLE)
    return emol.GetMol()


def connect_ligand(core, ligand, NH3_flag=False):
    """
    Function that takes two mol objects at creates a core with ligand.
    Args:
        core (Mol): The core to put ligand on. With dummy atoms at
            ligand positions.
        ligand (Mol): The ligand to put on core with dummy atom where
            N from the core should be.
        NH3_flag (bool): Flag to mark if the core has a NH3 on it.
            Then the charge is set to ensure non-faulty sanitation.
    Returns:
        mol (Mol): mol object with the ligand put on the core.
    """

    # mol object for dummy atom to replace on the core
    dummy = Chem.MolFromSmiles("*")

    # Get the dummy atom and then its atom object
    dummy_idx = ligand.GetSubstructMatch(Chem.MolFromSmiles("*"))
    atom = ligand.GetAtomWithIdx(dummy_idx[0])

    # Get neighbors to the dummy atom. Should be only 1 neighbor
    neighbor_pairs = [(dummy_idx[0], x.GetIdx()) for x in atom.GetNeighbors()]

    # Get the bond between dummy and neighbor and fragment on bond
    # To get ligand with a free bond that can attach to core
    new_bond = ligand.GetBondBetweenAtoms(
        neighbor_pairs[0][0], neighbor_pairs[0][1]
    ).GetIdx()
    frag = Chem.FragmentOnBonds(ligand, [new_bond], addDummies=False)
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # TODO IS THIS THE REASON FOR EMBEDDING ERROR? MAYBE BECAUSE IT PUTS DUMMY
    # ON CORE INSTEAD?
    # Put the ligand won the core with specified bonding atom in the ligand.
    mol = AllChem.ReplaceSubstructs(
        core,
        dummy,
        frags[0],
        replaceAll=True,
        replacementConnectionPoint=neighbor_pairs[0][1],
    )[0]

    # If NH3 is on the core, then the charge of NH3 must be set
    # to avoid sanitation error.
    if NH3_flag:
        mol.GetAtomWithIdx(23).SetFormalCharge(1)

    # Sanitation ensures that it is a reasonable molecule.
    Chem.SanitizeMol(mol)

    # If this is not done, the ligand i placed in zero.
    # This command removes the 3D coordinates of the core such
    # that the 2D mol is displayed nicely.
    mol.RemoveAllConformers()

    return mol


def connectMols(core, NH3_flag=True):

    query = Chem.MolFromSmarts("[NX3;H3]")
    mol = Chem.MolFromSmiles("[NH3]")
    if NH3_flag:
        mol.GetAtomWithIdx(0).SetFormalCharge(1)

    combined = Chem.CombineMols(core, mol)
    emol = Chem.EditableMol(combined)

    # Get the idx of the Mo and NH3
    NH3_match = combined.GetSubstructMatch(query)
    atom = combined.GetAtomWithIdx(NH3_match[0])
    NH3_idx = atom.GetIdx()

    for atom in combined.GetAtoms():
        if atom.GetAtomicNum() == 42:
            Mo_idx = atom.GetIdx()
            break

    emol.AddBond(Mo_idx, NH3_idx, order=Chem.rdchem.BondType.SINGLE)
    mol = emol.GetMol()

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

    # Look for teriary amines in the input ligand.
    tert_amines = ligand.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H0;D3;!+1]"))
    if len(tert_amines) == 0:
        raise Exception(
            f"{Chem.MolToSmiles(Chem.RemoveHs(ligand))} constains no tertiary amine."
        )

    # Loop over found amines
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

        # Initilize dummy pattern
        patt = Chem.MolFromSmarts("[1*]")

        # Get list of ligands with only one dummy atom.
        # this also excludes the remaining tertiary amin which will have 3 dummies
        ligands = [
            struct for struct in frags if len(struct.GetSubstructMatches(patt)) == 1
        ]

        # Initialize primary amine
        NH2_mol = Chem.MolFromSmiles("[NH2]")

        # If all N ligands give a viable ligand, we have to
        # randomly choose one of them
        ligand = random.choice(ligands)

        # Replace the dummy on the ligand with a primary amine.
        ligand = AllChem.ReplaceSubstructs(
            ligand, dummy, NH2_mol, replacementConnectionPoint=0
        )[0]

        # Little hack to remove the dot (open bond) on NH2 when visualizing the new ligand.
        lig = Chem.MolFromSmiles(Chem.MolToSmiles(ligand))

        # If there is a valid ligand break the for loop
        if ligands:
            break

    return lig


# DEPRECATED
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
    # Initialize patternhttps://github.com/Strandgaard96/GB_GA.git
    patt = Chem.MolFromSmarts(smart)

    # Get the ligand that is not NH2
    ligands = [struct for struct in frags if len(struct.GetSubstructMatches(patt)) == 0]

    return ligands


def create_prim_amine(input_ligand):
    """
    A function that takes a ligand and splits on a nitrogen bond, and then gives
    a ligand out that has an NH2 and a cut_idx that specifies the location of the primary
    amine and where to cut when putting ligand putting on the Mo core
    Args:
        input_ligand (mol): A regular ligand with no dummy atoms.

    Returns:
        output_ligand (mol): Modified ligand with an added primary amine
        prim_amine_index[0] tuple(int): idx of the primary amine
    """

    # Initialize dummy mol
    dummy = Chem.MolFromSmiles("*")

    # Add explicit hydrogens to the molecule
    input_ligand = Chem.AddHs(input_ligand)

    # Match Secondary or Tertiary amines
    matches = input_ligand.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H1,H0;!+1]"))
    if len(matches) == 0:
        raise Exception(
            f"{Chem.MolToSmiles(Chem.RemoveHs(input_ligand))} constains no amines to split on"
        )

    # TODO perhaps randomly scramble the matches list

    # Randomly select one of the amines.
    for match in matches:

        # Get the neigbouring bonds to the selected amine
        atom = input_ligand.GetAtomWithIdx(match[0])

        # Create list of tuples that contain the amine idx and idx of each of the three
        # neighbors that are not hydrogne.
        indices = [
            (match[0], x.GetIdx())
            for x in atom.GetNeighbors()
            if (
                (x.GetAtomicNum() != 1)
                and not input_ligand.GetBondBetweenAtoms(
                    match[0], x.GetIdx()
                ).IsInRing()
            )
        ]

        bond = []
        if indices:
            break

    # TODO Add try statement here if indices is empty
    try:
        atoms = random.choice(indices)
    except IndexError as e:
        print("Oh no, found no valid cut points")
        return input_ligand, None
    bond.append(input_ligand.GetBondBetweenAtoms(*atoms).GetIdx())

    # Get the fragments from breaking the amine bonds.
    # OBS! If the fragments connected to the tertiary amine, are connected
    # then the resulting ligand will have multiple dummy locations which will break
    # the workflow
    frag = Chem.FragmentOnBonds(
        input_ligand, bond, addDummies=True, dummyLabels=[(1, 1)]
    )
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # Select the fragments that are not the amine the ligand was cut from.
    # If there is only one fragment, it can break so i added the temporary
    # If statement
    # TODO: handle this better
    if len(frags) == 1:
        ligand = [frags[0]]
    else:
        ligand = [
            struct
            for struct in frags
            if len(struct.GetSubstructMatches(Chem.MolFromSmarts("[1*][N]"))) == 0
        ]

    # As this function is also run outside paralellization, an error here will break
    # the whole driver. This statement ensures that something is returned at least
    # If the list is empty.
    if not ligand:
        return input_ligand, None

    # Put primary amine on the dummy location for the ligand just created.
    # Currently only the first ligand is selected.
    NH2_mol = Chem.MolFromSmiles("[NH2]")
    lig = AllChem.ReplaceSubstructs(
        ligand[0], dummy, NH2_mol, replacementConnectionPoint=0, replaceAll=True
    )[0]
    # Small hack to prevent the dot open bond on NH2.
    output_ligand = Chem.MolFromSmiles(Chem.MolToSmiles(lig))

    # Get idx where to cut and we just return of of them.
    prim_amine_index = output_ligand.GetSubstructMatches(
        Chem.MolFromSmarts("[NX3;H2;!+1]")
    )
    if len(prim_amine_index) > 1:
        print(
            f"There are several primary amines to cut at with idxs: {prim_amine_index}"
            f"changing one to hydrogen"
        )
        # Replace dummy with hydrogen in the frag:
        output_ligand = AllChem.ReplaceSubstructs(
            output_ligand,
            Chem.MolFromSmarts("[NX3;H2;!+1]"),
            Chem.MolFromSmiles("[H]"),
            replacementConnectionPoint=0,
        )[0]
        prim_amine_index = output_ligand.GetSubstructMatches(
            Chem.MolFromSmarts("[NX3;H2;!+1]")
        )
    return output_ligand, prim_amine_index


def create_dummy_ligand(ligand, cut_idx=None):
    """
    Takes mol object and splits it based on a primary amine such that the frags can connect to
    the tertiary amine on the Mo core.
    Args:
        cut_idx tuple(int):
        ligand (mol):
    Returns:
        ligands List(mol) :
    """
    # TODO AllChem.ReplaceCore() could be used here instead

    # Initialize dummy mol
    dummy = Chem.MolFromSmiles("*")

    # Create explicit hydrogens and sterechemistry i dont know what does.
    ligand = Chem.AddHs(ligand)

    # Get the neigbouring bonds to the amine given by cut_idx
    atom = ligand.GetAtomWithIdx(cut_idx)

    # Create list of tuples that contain the amine idx and idx of neighbor.
    indices = [
        (cut_idx, x.GetIdx()) for x in atom.GetNeighbors() if x.GetAtomicNum() != 1
    ][0]

    # Get the bonds to the neighbors.
    bond = []
    bond.append(ligand.GetBondBetweenAtoms(indices[0], indices[1]).GetIdx())

    # Get the two fragments, the ligand and the NH2
    frag = Chem.FragmentOnBonds(ligand, bond, addDummies=True, dummyLabels=[(1, 1)])
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    # Pattern for NH2+dummy
    smart = "[1*][N]([H])([H])"
    patt = Chem.MolFromSmarts(smart)

    # Get the ligand that is not NH2
    ligands = [struct for struct in frags if len(struct.GetSubstructMatches(patt)) == 0]

    return ligands[0]


def embed_rdkit(
    mol,
    core,
    numConfs=1,
    coreConfId=-1,
    randomseed=2342,
    numThreads=1,
    force_constant=1e3,
    pruneRmsThresh=1,
):
    """Embedding driver function

    Args:
        mol (Mol): Core+ligand
        core (Mol): Core with dummy atoms on ligand positions
        numConfs (int): How many conformers to get from embedding
        coreConfId (int): If core has multiple conformers this indicates which one to choose
        randomseed (int): Seed for embedding.
        numThreads (int): How many threads to use for embedding.
        force_constant (float): For alignment
        pruneRmsThresh (int): Embedding parameter

    Returns:

    """

    # Match the core+ligand to the Mo core.
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    sio = sys.stderr = StringIO()
    # if not AllChem.UFFHasAllMoleculeParams(mol):
    #    raise Exception(Chem.MolToSmiles(mol), sio.getvalue())

    # Get the coordinates for the core, which constrains the embedding
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    # The random seed might be irrelevant here as randomcoords are false
    cids = AllChem.EmbedMultipleConfs(
        mol=mol,
        numConfs=numConfs,
        coordMap=coordMap,
        maxAttempts=10,
        randomSeed=2,
        numThreads=numThreads,
        pruneRmsThresh=pruneRmsThresh,
        useRandomCoords=False,
    )
    Chem.SanitizeMol(mol)

    cids = list(cids)
    if len(cids) == 0:
        # Retry with a different random seed
        cids = AllChem.EmbedMultipleConfs(
            mol=mol,
            numConfs=numConfs,
            coordMap=coordMap,
            maxAttempts=10,
            randomSeed=random.randint(0, 2048),
            numThreads=numThreads,
            pruneRmsThresh=pruneRmsThresh,
            useRandomCoords=True,
        )
        Chem.SanitizeMol(mol)
        if len(cids) == 0:
            print(coordMap, Chem.MolToSmiles(mol))
        raise ValueError("Could not embed molecule")

    # TODO is this step necessarry for me?
    # Rotate embedded conformations onto the core
    algMap = [(j, i) for i, j in enumerate(match)]
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
