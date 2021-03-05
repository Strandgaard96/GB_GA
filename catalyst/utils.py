# %%
import py3Dmol
from IPython.display import display

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToMolFile
from rdkit.Chem import GetPeriodicTable

import numpy as np
import time
import shutil
import os
import sys
sys.path.append('/home/julius/soft/')
from xyz2mol.xyz2mol import read_xyz_file, xyz2mol
sys.path.append('/home/julius/soft/GB-GA')
# from catalyst.gaussian_utils import extract_optimized_structure

# %%

def draw3d(mols, width=600, height=600, Hs=True, confId=-1, multipleConfs=False,atomlabel=False):
    try:
        p = py3Dmol.view(width=width,height=height)
        if type(mols) is not list:
            mols = [mols]
        for mol in mols:
            if multipleConfs:
                for conf in mol.GetConformers():
                    mb = Chem.MolToMolBlock(mol, confId=conf.GetId())
                    p.addModel(mb,'sdf')
            else:
                if type(mol) is str:
                    if os.path.splitext(mol)[-1] == '.xyz':
                        xyz_f = open(mol)
                        line = xyz_f.read()
                        xyz_f.close()
                        p.addModel(line,'xyz')   
                    # elif os.path.splitext(mol)[-1] == '.out':
                    #     xyz_file = extract_optimized_structure(mol, return_mol=False)
                    #     xyz_f = open(xyz_file)
                    #     line = xyz_f.read()
                    #     xyz_f.close()
                    #     p.addModel(line,'xyz') 
                else:           
                    mb = Chem.MolToMolBlock(mol, confId=confId)
                    p.addModel(mb,'sdf')
        p.setStyle({'sphere':{'radius':0.4}, 'stick':{}})
        if atomlabel:
            p.addPropertyLabels('index')#,{'elem':'H'}
        p.zoomTo()
        p.update()
        # p.show()
    except:
        print('py3Dmol, RDKit, and IPython are required for this feature.')

def vis_trajectory(xyz_file,atomlabel=False):
    try:
        xyz_f = open(xyz_file)
        line = xyz_f.read()
        xyz_f.close()
        p = py3Dmol.view(width=400,height=400)
        p.addModelsAsFrames(line,'xyz')
        # p.setStyle({'stick':{}})
        p.setStyle({'sphere':{'radius':0.4}, 'stick':{}})
        if atomlabel:
            p.addPropertyLabels('index',{'elem':'H'})
        p.animate({'loop': "forward",'reps': 10})
        p.zoomTo()
        p.show()
    except:
        raise

def mol_from_xyz(xyz_file, charge=0):
    atoms, _, xyz_coordinates = read_xyz_file(xyz_file)
    mol = xyz2mol(atoms, xyz_coordinates, charge)
    return mol

def sdf2mol(sdf_file):
    mol = Chem.SDMolSupplier(sdf_file,removeHs = False, sanitize=True)[0]
    return mol

def mols_from_smi_file(smi_file, n_mols=None):
    mols = []
    with open(smi_file) as _file:
        for i, line in enumerate(_file):
            mols.append(Chem.MolFromSmiles(line))
            if n_mols:
                if i == n_mols-1:
                    break
    return mols

def mkdir(directory, overwrite=False):
    if os.path.exists(directory) and overwrite:
        shutil.rmtree(directory)
    os.mkdir(directory)

def hartree2kcalmol(hartree):
    return hartree * 627.5095

def hartree2kJmol(hartree):
    return hartree * 2625.50 

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self, text="{:0.4f} seconds", logger=print):
        self._start_time = None
        self.text = text
        self.logger = logger

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        return elapsed_time



from dataclasses import dataclass, field
from typing import List
import pickle
from tabulate import tabulate
import copy

@dataclass
class Individual:
    rdkit_mol:          Chem.rdchem.Mol = field(repr=False, compare=False)
    smiles:             str = field(init=False, compare=True)
    idx:                tuple = field(default=(None,None), repr=False, compare=False)
    score:              float = field(default=None, repr=False, compare=False)
    energy:             float = field(default=None, repr=False, compare=False)
    sa_score:           float = field(default=None, repr=False, compare=False)
    normalized_fitness: float = field(default=None, repr=False, compare=False)
    neutral_rdkit_mol:  Chem.rdchem.Mol = field(init=False, repr=False, compare=False)
    warnings:           List[str] = field(default_factory=list, repr=False, compare=False) 
    # parents:            

    def __post_init__(self):
        self.smiles = Chem.MolToSmiles(self.rdkit_mol)

    def update(self):
        if self.energy and self.sa_score:
            setattr(self, 'score', -self.energy*self.sa_score)
    
    def list_of_props(self):
        return([self.idx[1], self.score, self.energy, self.sa_score, self.smiles])

@dataclass(order=True)
class Population():
    generation_num:     int = field(default=None, init=True)
    molecules:          List[Individual] = field(default_factory=list) 

    # def __post_init__(self):
    #     for i, molecule in enumerate(self.molecules):
    #         setattr(molecule, 'idx', (self.generation_num, i))

    def __add__(self, other):
        return Population(molecules = self.molecules + other.molecules)

    def assign_idx(self):
        for i, molecule in enumerate(self.molecules):
            setattr(molecule, 'idx', (self.generation_num, i))
    
    def update(self):
        for molecule in self.molecules:
            molecule.update()

    def get(self, prop):
        properties = []
        for molecule in self.molecules:
            properties.append(getattr(molecule, prop))
        return properties

    def setprop(self, prop, list_of_values):
        for molecule, value in zip(self.molecules, list_of_values):
            setattr(molecule, prop, value)

    def sortby(self, prop, reverse=True):
        self.molecules.sort(key=lambda x: getattr(x,prop), reverse=reverse)

    def prune(self, population_size):
        self.sortby('score')
        self.molecules = self.molecules[:population_size]

    def safe(self, directory):
        filename = 'GA_output.pkl'
        with open(filename, 'ab+') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def print(self):
        table = []
        for individual in self.molecules:
            table.append(individual.list_of_props())
        if isinstance(self.generation_num, int):
            print(f'\nGeneration {self.generation_num:02d}')
        print(tabulate(table, headers=['idx', 'score', 'energy', 'sa_score', 'smiles']))



# %%
# i1 = Individual(Chem.MolFromSmiles('CO'))
# i2 = Individual(Chem.MolFromSmiles('O'))
# i3 = Individual(Chem.MolFromSmiles('C'))
# i4 = Individual(Chem.MolFromSmiles('C'))


# # %%
# pop = Population(generation_num=1, molecules=[i1,i2,i3])
# pop.print()
# %%

# %%
