# -*- coding: utf-8 -*-
"""
Module that contains mol manipulations and various resuable functionality classes.

Todo:
    * Refactor functionality
"""
from collections import UserDict
import os, sys

from rdkit import Chem
from rdkit.Chem import AllChem

# from rdkit.Chem import Draw


class DotDict(UserDict):
    """dot.notation access to dictionary attributes
    Currently not in use as it clashed with Multiprocessing-Pool pickling"""

    __getattr__ = UserDict.get
    __setattr__ = UserDict.__setitem__
    __delattr__ = UserDict.__delitem__


class cd:
    """Context manager for changing the current working directory dynamically.
    # See: https://book.pythontips.com/en/latest/context_managers.html"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        # Print traceback if anything happens
        if traceback:
            print(sys.exc_info())
        os.chdir(self.savedPath)


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

    # List to hold the various core+ligand structures created
    mols = []
    dummy = Chem.MolFromSmiles("*")
    # Take the original ligand from the zinc dataset and split it based on
    # tertiary amines. Return a list containing the ligands.
    ligands = create_ligands(ligand)

    # Iterate through the ligands and put them on the provided core
    for i, ligand in enumerate(ligands):

        # Now we need to get the indice of the atom the dummy is bound to and
        # remove the dummy atom while keeping the idx of the atom it was bound to
        dummy_idx = ligand.GetSubstructMatch(Chem.MolFromSmiles("*"))
        neigh = ligand.GetAtomWithIdx(dummy_idx[0])
        bond = [(dummy_idx[0], x.GetIdx()) for x in neigh.GetNeighbors()]
        new_bond = ligand.GetBondBetweenAtoms(bond[0][0], bond[0][1]).GetIdx()
        frag = Chem.FragmentOnBonds(ligand, [new_bond], addDummies=False)
        frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

        # Put the ligand won the core with specified bonding atom in the ligand.
        mol = AllChem.ReplaceSubstructs(
            core,
            dummy,
            frags[0],
            replaceAll=True,
            replacementConnectionPoint=bond[0][1],
        )[0]

        if NH3_flag:
            mol.GetAtomWithIdx(23).SetFormalCharge(1)
        # Sanitation ensures that it is a reasonable molecule.
        Chem.SanitizeMol(mol)
        # If this is not done, the ligand i placed in zero. Dunno why.
        mol.RemoveAllConformers()
        mols.append(mol)

    # Show final result for debug
    # img = Draw.MolsToImage(mols)
    # img.show()
    return mols


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

        # If valid ligands were found, break for loop and return ligands
        if ligands:
            break

    save_ligand_smiles = False
    if save_ligand_smiles:
        with open("test_ligand.smi", "w") as f:
            f.write(Chem.MolToSmiles(Chem.RemoveHs(ligands[0])))

    return ligands
