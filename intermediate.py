import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from glob import glob
import pickle
import shutil
from rdkit import Chem
from rdkit.Chem import Draw

def main():

    # Load intermediate
    mol = Chem.MolFromMolFile(
        "/home/magstr/Documents/nitrogenase/schrock/diagrams_schrock/schrock_dft/Mo_N2/ams.results/traj.mol", sanitize = False, removeHs=False)

    # Initialize substructure string

    # : matches aromatic bonds. Not == when it is delocalized
    # '[c]:[c][N]'
    # Smart for a nitrogen bound to aromatic carbon.
    smart = '[c][N]'

    # Initialize pattern
    patt = Chem.MolFromSmarts(smart)

    # Substructure match
    print(f'Has substructure match: {mol.HasSubstructMatch(patt)}')
    indices = mol.GetSubstructMatches(patt)

    # Visualize
    #im = Chem.Draw.MolToImage(mol, size=(800, 800))
    #im.show()

    bonds=[]
    # Cut the bonds between the nitrogen and the carbon.
    for tuple in indices:
        # Get bond number
        bonds.append(mol.GetBondBetweenAtoms(tuple[0], tuple[1]).GetIdx())

    # Cut
    frag = Chem.FragmentOnBonds(mol, bonds,addDummies=True,dummyLabels=[(1, 1),(1, 1),(1, 1)])

    #Draw cut
    #Chem.Draw.MolToImage(frag, size=(800, 800))

    # Split mol object into individual fragments. sanitizeFrags needs to be false, otherwise not work.
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    #Show a specific frag
    im = Chem.Draw.MolToImage(frags[1], size=(800, 800))
    im.show()

    # Command for drawing all frags in one
    im = Draw.MolsToGridImage(frags)
    im.show()

    # Save frag to file
    with open('frag.mol', 'w+') as f:
        f.write(Chem.MolToMolBlock(frags[1]))




if __name__ == '__main__':
    main()