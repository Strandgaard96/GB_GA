import math
import os
import pickle
import time
from typing import List

from catalystGA import BaseCatalyst
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


class Schrock(BaseCatalyst):

    n_ligands = 4

    def __init__(self, metal: Chem.Mol, ligands: List):
        super().__init__(metal, ligands)

    def calculate_score(self, ncores, envvar_scratch):
        start = time.time()
        try:
            self.assemble()
            m = Chem.AddHs(self.mol)
            cid = AllChem.EmbedMolecule(m, useRandomCoords=True)
            if cid != 0:
                raise Exception("Embedding failed")
            logP = Descriptors.MolLogP(self.m)
        except Exception as e:
            self.error = str(e)
            logP = math.nan

        self.embmol = m
        self.score = logP
        self.timing = time.time() - start
        self.save()

    def save(self, directory="."):
        """Dump ind object into file."""
        filename = os.path.join(directory, f"ind.pkl")
        with open(filename, "ab+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
