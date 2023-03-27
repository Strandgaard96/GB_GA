import math
import time
from typing import List

from catalystGA import BaseCatalyst
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


class Schrock(BaseCatalyst):

    n_ligands = 4

    def __init__(self, metal: Chem.Mol, ligands: List):
        super().__init__(metal, ligands)

    def calculate_score(self):
        start = time.time()
        try:
            self.assemble()
            cid = AllChem.EmbedMolecule(self.mol, useRandomCoords=True)
            if cid != 0:
                raise Exception("Embedding failed")
            logP = Descriptors.MolLogP(self.mol)
        except Exception as e:
            self.error = str(e)
            logP = math.nan

        self.score = logP
        self.timing = time.time() - start
