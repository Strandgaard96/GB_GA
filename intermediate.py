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
        "/home/magstr/Documents/nitrogenase/schrock/diagrams_schrock/schrock_dft/Mo/ams.results/traj.mol")

    # Initialize substructure string

    # : matches aromatic bonds. Not == when it is delocalized
    # '[c]:[c][N]'

    # Smart for a nitrogen bound to aromatic carbon.
    smart = '[c][N]'

    patt = Chem.MolFromSmarts(smart)
    a = Draw.MolToImage(mol, size=(800, 800))
    a.show()
    print(Chem.MolToSmiles(patt), Chem.MolToSmiles(mol))
    # Get indice of the carbon atom
    print(f'Has substructure match: {mol.HasSubstructMatch(patt)}')
    indices = mol.GetSubstructMatches(patt)
    print(indices)



if __name__ == '__main__':
    main()