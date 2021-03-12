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
import pandas as pd




# # %%
# import matplotlib.pyplot as plt

# test = np.linspace(-4,4)
# out = []
# for val in test:
#     out.append(scaling_function(val, method='other'))
# plt.plot(test,out)

# %%
# %%

@dataclass
class Individual:
    rdkit_mol:          Chem.rdchem.Mol = field(repr=False, compare=False)
    smiles:             str = field(init=False, compare=True, repr=True)
    idx:                tuple = field(default=(None,None), repr=False, compare=False)
    parentA_idx:        tuple = field(default=None, repr=False, compare=False)
    parentB_idx:        tuple = field(default=None, repr=False, compare=False)
    survival_idx:       tuple = field(default=None, repr=False, compare=False)
    score:              float = field(default=None, repr=False, compare=False)
    energy:             float = field(default=None, repr=False, compare=False)
    sa_score:           float = field(default=None, repr=False, compare=False)
    normalized_fitness: float = field(default=None, repr=False, compare=False)
    neutral_rdkit_mol:  Chem.rdchem.Mol = field(init=False, repr=False, compare=False)
    warnings:           List[str] = field(default_factory=list, repr=False, compare=False) 

    def __post_init__(self):
        self.smiles = Chem.MolToSmiles(self.rdkit_mol)
        num_parents = 0
        if self.parentA_idx:
            num_parents += 1
        if self.parentB_idx:
            num_parents += 1
        self.num_parents = num_parents

    # def update(self, method='linear', from_min=-3, from_max=1, a=2, b=-2):
    #     if self.energy and self.sa_score:
    #         setattr(self, 'score', scaling_function(self.energy, method=method, from_min=from_min, from_max=from_max, a=a, b=b)*self.sa_score)

    def list_of_props(self):
        return([self.idx[1], self.normalized_fitness, self.score, self.energy, self.sa_score, self.smiles])

@dataclass(order=True)
class Population:
    generation_num:     int = field(default=None, init=True)
    molecules:          List[Individual] = field(repr=False,default_factory=list) 
    size:               int = field(default=None, init=True)

    def __post_init__(self):
        self.size = len(self.molecules)

    def assign_idx(self):
        for i, molecule in enumerate(self.molecules):
            setattr(molecule, 'idx', (self.generation_num, i))
        self.size = len(self.molecules)
    
    # def update(self):
    #     for molecule in self.molecules:
    #         molecule.update()

    def get(self, prop):
        properties = []
        for molecule in self.molecules:
            properties.append(getattr(molecule, prop))
        return properties

    def setprop(self, prop, list_of_values):
        for molecule, value in zip(self.molecules, list_of_values):
            setattr(molecule, prop, value)

    def appendprop(self, prop, list_of_values):
        for molecule, value in zip(self.molecules, list_of_values):
            if value:
                getattr(molecule, prop).append(value)

    def sortby(self, prop, reverse=True):
        if reverse:
            self.molecules.sort(key=lambda x: (getattr(x,prop) is not None, getattr(x,prop)), reverse=reverse)
        else:
            self.molecules.sort(key=lambda x: (getattr(x,prop) is None, getattr(x,prop)), reverse=reverse)

    def prune(self, population_size):
        self.sortby('score', reverse=True)
        self.molecules = self.molecules[:population_size]
        self.size = len(self.molecules)

    def print(self):
        table = []
        for individual in self.molecules:
            table.append(individual.list_of_props())
        if isinstance(self.generation_num, int):
            print(f'\nGeneration {self.generation_num:02d}')
        print(tabulate(table, headers=['idx', 'normalized_fitness', 'score', 'energy', 'sa_score', 'smiles']))

    def pop2pd(self, columns = ['score', 'energy', 'sa_score', 'rdkit_mol']):
        df = pd.DataFrame(list(map(list, zip(*[self.get(prop) for prop in columns]))), index=pd.MultiIndex.from_tuples(self.get('idx'), names=('generation', 'individual')))
        df.columns = columns
        return df

@dataclass(order=True)
class Generation:
    generation_num:     int = field(init=True) 
    children:           Population= field(default_factory=Population) 
    survivors:          Population = field(default_factory=Population)
    # a counter to count how ofter molecules got flagged during mutation/crossover by filter

    def __post_init__(self):
        if self.generation_num != self.children.generation_num or self.generation_num != self.survivors.generation_num:
            raise Warning(f'Generation {self.generation_num} has Children from generation {self.children.generation_num} and survivors from generation {self.survivors.generation_num}')

    def save(self, directory):
        filename = os.path.join(directory, 'GA_output.pkl')
        with open(filename, 'ab+') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def print(self, population='survivors'):
        if population == 'survivors':
            population = self.survivors
        elif population == 'children':
            population = self.children
        else:
            raise TypeError(f'{population} is not a valid option (survivors/children)')
        table = []
        for individual in population.molecules:
            table.append(individual.list_of_props())
        print(f'\nGeneration {self.generation_num:02d}')
        print(tabulate(table, headers=['idx', 'normalized_fitness', 'score', 'energy', 'sa_score', 'smiles']))
    
    def gen2pd(self, population='survivors', columns = ['score', 'energy', 'sa_score', 'rdkit_mol']):
        if population == 'survivors':
            population = self.survivors
        elif population == 'children':
            population = self.children
        else:
            raise TypeError(f'{population} is not a valid option (survivors/children)')
        pd = population.pop2pd(columns=columns)
        return pd

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(generation_num={self.generation_num!r}, children_size={self.children.size}, survivors_size={self.survivors.size})')

@dataclass(order=True)
class GA_run:
    num_generations:        int = field(default=None, init=True)
    generations:            List[Generation] = field(default_factory=list)

    def __post_init__(self):
        self.num_generations = len(self.generations)

    def append_gen(self, generation):
        self.generations += [generation]
        self.num_generations = len(self.generations)

    def get_ind(self, idx):
        return self.generations[idx[0]].survivors.molecules[idx[1]]

    def get_parent_tuple(self, individual):
        if not individual.parentA_idx and not individual.parentB_idx:
            return None
        if individual.parentA_idx:
            parentA = self.get_ind(individual.parentA_idx)
        if individual.parentB_idx:
            parentB = self.get_ind(individual.parentB_idx)
        return tuple((individual.idx, parentA.idx, parentB.idx))

    def get_survivor(self, individual):
        if not individual.survival_idx:
            return None
        else:
            return tuple((individual.idx, individual.survival_idx))

    def get_origin(self, individual):
        if individual.survival_idx:
            return self.get_survivor(individual)
        elif individual.parentA_idx:
            return self.get_parent_tuple(individual)

    def traceback(self, idx, max_its=100):
        trace = []
        checked = []
        youngest = self.get_ind(idx)
        trace.append(self.get_origin(youngest))
        counter = 0
        for node in trace:
            if not node:
                break
            iD = node[0]
            if iD in checked:
                continue # with next node
            trace.append(self.get_origin(self.get_ind(node[1])))
            if len(node) == 3:
                trace.append(self.get_origin(self.get_ind(node[2])))
            counter += 1
            if counter > max_its:
                break
        return [x for x in trace if x is not None]

                


    def print(self, population='survivors'):
        for generation in self.generations:
            generation.print(population)

    def ga2pd(self, population='survivors', columns = ['score', 'energy', 'sa_score', 'rdkit_mol']):
        df = pd.concat([generation.gen2pd(population, columns=columns) for generation in self.generations])
        return df

def load_GA(pkl):
    with open(pkl, "rb") as rfp:
        ga = GA_run()
        while True:
            try:
                generation = pickle.load(rfp)
                ga.append_gen(generation)
            except EOFError:
                break
    return ga
# %%

if __name__ == '__main__':
    import sys
    sys.path.append('/home/julius/soft/GB-GA/')
    import GB_GA as ga 

    def load_GA(pkl):
        with open(pkl, "rb") as rfp:
            ga = GA_run()
            while True:
                try:
                    generation = pickle.load(rfp)
                    ga.append_gen(generation)
                except EOFError:
                    break
        return ga

# %%
    ga = load_GA('/home/julius/thesis/data/ZINC_scoring/ts_scoring/GA_output.pkl')
    
# %%
    ind1 = Individual(Chem.MolFromSmiles('CCC'), energy=100, score=1)
    ind2 = Individual(Chem.MolFromSmiles('CNC'), energy=None, score=None)
    ind3 = Individual(Chem.MolFromSmiles('COC'), energy=-2, score=10)

    pop = Population(generation_num=0, molecules=[ind1,ind2,ind3])
    pop.assign_idx()

# %%
