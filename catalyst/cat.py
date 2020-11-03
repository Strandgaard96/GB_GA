from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from scoring_functions import shell, write_xtb_input_file

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import numpy as np
import os
import shutil
import string
import random
import time

seed=101

def get_structure(start_mol,n_confs):
	mol = Chem.AddHs(start_mol)
	new_mol = Chem.Mol(mol)

	confIDs = AllChem.EmbedMultipleConfs(mol,numConfs=n_confs,useExpTorsionAnglePrefs=True,useBasicKnowledge=True,maxAttempts=5000)#,randomSeed=seed)
	energies = AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters=2000, nonBondedThresh=100.0)

	energies_list = [e[1] for e in energies]
	min_e_index = energies_list.index(min(energies_list))

	new_mol.AddConformer(mol.GetConformer(min_e_index))

	return new_mol

def get_xtb_energy(out):
	'''	Returns electronic energy calculated by xTB	'''
	if 'total energy' in str(out):
		xtb_energy = float(str(out).split('total energy')[1].split('Eh')[0])
	else:
		xtb_energy = 0
	return xtb_energy

def compute_energy(mol,n_confs):
	'''	Generates n_confs conformers from input mol, starts xTB calculation with gbsa MeOH solvent model and returns electronic energy '''
	mol = get_structure(mol,n_confs)
	dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
	os.mkdir(dir)
	os.chdir(dir)
	write_xtb_input_file(mol, 'test')
	out = shell('xtb test+0.xyz --opt --gbsa methanol',shell=False)
	energy = get_xtb_energy(out)
	os.chdir('..')
	shutil.rmtree(dir)
	return energy

def connect_amine_with_struc(rdkit_mol, structure_smi='[H]c1c([H])c([C@]([H])([O-])[C@@]([H])(C(=O)OC([H])([H])[H])C([H])([H])[N@+]23C([H])([H])C([H])([H])[C@]([H])(C([H])([H])C2([H])[H])C([H])([H])C3([H])[H])c([H])c([H])c1[N+](=O)[O-]'):
	'''	Takes mol that contains amine and connects it to 5-sr, returns list of 6*n_amine products (n_amine = number of amine substructe in input mol)	'''
	struc = Chem.MolFromSmiles(structure_smi)
	connect_smarts = '[#7X3;H0;D3;!+1]([*:1])([*:2])[*:3].[C$([*]([#7X4;H0;D4&+1])([CH1])):4][#7X4;H0;D4&+1]>>[#7X4;H0;D4&+1]([*:1])([*:2])([*:3])[C:4]'
	rxn = AllChem.ReactionFromSmarts(connect_smarts)
	ps = rxn.RunReactants((rdkit_mol,struc))
	return ps

def number_of_conformers(mol):
	n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
	n_confs = 5 + 5*n_rot
	return n_confs

def compute_energy_diff(rdkit_mol, n_confs=None):
	'''	Takes mol containing amine and calculates its energy as well as the energy of amine+5-st molecule and returns energy difference 	'''	
	start = time.time()
	if n_confs is None:
		n_confs = number_of_conformers(rdkit_mol)
	e_cat = compute_energy(rdkit_mol,n_confs)
	products = connect_amine_with_struc(rdkit_mol)
	unique = get_unique_amine_products(rdkit_mol,products)
	energy_of_conformers = []
	for conf in unique:
		conf = cleanup(conf)
		energy_of_conformers.append(compute_energy(conf,n_confs))
	min_e_conf = min(energy_of_conformers)
	energy_diff = (-52.277827212938 + e_cat) - min_e_conf
	# print(f'{Chem.MolToSmiles(rdkit_mol)} \n\tConformers: {n_confs} \n\tUnique products: {len(unique)} \n\tEnergy Difference: {energy_diff} \n\tDuration: {time.time()- start:.2f} s')
	return energy_diff

def energydiff2score(energy):
	score = -energy 
	return score

def cat_scoring(rdkit_mol, n_confs=None):
	energy_diff = compute_energy_diff(rdkit_mol, n_confs=n_confs)
	score = energydiff2score(energy_diff)
	return score

def count_substruc_match(mol, pattern):
	'''	Counts how ofter a pattern occurs in mol, returns count	'''
	count = len(mol.GetSubstructMatches(pattern))
	return count

def get_unique_amine_products(rdkit_mol, products):
	'''	Returns unique products from connect_amine_with_struc '''
	tert_amine = Chem.MolFromSmarts('[#7X3;H0;D3;!+1]')
	quart_amine = Chem.MolFromSmarts('[#7X4;H0;D4&+1]')
	n = count_substruc_match(rdkit_mol, tert_amine)
	unique = []
	for n in np.arange(n):
		unique.append(products[n*6][0])
	return unique

def cleanup(mol):
	'''	Returns nice 2D mol '''
	Chem.SanitizeMol(mol)
	smi = Chem.MolToSmiles(mol)
	new_mol = Chem.MolFromSmiles(smi)
	return new_mol