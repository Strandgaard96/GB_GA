'''
Written by Jan H. Jensen 2018
'''

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import rdFMCS

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import numpy as np
import sys
from multiprocessing import Pool
import subprocess
import os
import shutil
import string
import random

import sascorer

logP_values = np.loadtxt('logP_values.txt')
SA_scores = np.loadtxt('SA_scores.txt')
cycle_scores = np.loadtxt('cycle_scores.txt')
SA_mean =  np.mean(SA_scores)
SA_std=np.std(SA_scores)
logP_mean = np.mean(logP_values)
logP_std= np.std(logP_values)
cycle_mean = np.mean(cycle_scores)
cycle_std=np.std(cycle_scores)

def calculate_score(args):
	'''Parallelize at the score level (not currently in use)'''
	gene, function, scoring_args = args
	score = function(gene,scoring_args)
	return score

def calculate_scores_parallel(population,function,scoring_args, n_cpus):
	'''Parallelize at the score level (not currently in use)'''
	args_list = []
	args = [function, scoring_args]
	for gene in population:
		args_list.append([gene]+args)
		
	with Pool(n_cpus) as pool:
		scores = pool.map(calculate_score, args_list)

	return scores

def calculate_scores(population,function,scoring_args):
	scores = []
	for gene in population:
		score = function(gene,scoring_args)
		scores.append(score)

	return scores 

def logP_max(m, dummy):
	score = logP_score(m)
	return max(0.0, score)

def logP_target(m,args):
	target, sigma = args
	score = logP_score(m)
	score = GaussianModifier(score, target, sigma)
	return score


def logP_score(m):
	try:
		logp = Descriptors.MolLogP(m)
	except:
		print (m, Chem.MolToSmiles(m))
		sys.exit('failed to make a molecule')

	SA_score = -sascorer.calculateScore(m)
	#cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(m)))
	cycle_list = m.GetRingInfo().AtomRings() #remove networkx dependence
	if len(cycle_list) == 0:
		cycle_length = 0
	else:
		cycle_length = max([ len(j) for j in cycle_list ])
	if cycle_length <= 6:
		cycle_length = 0
	else:
		cycle_length = cycle_length - 6
	cycle_score = -cycle_length
	#print cycle_score
	#print SA_score
	#print logp
	SA_score_norm=(SA_score-SA_mean)/SA_std
	logp_norm=(logp-logP_mean)/logP_std
	cycle_score_norm=(cycle_score-cycle_mean)/cycle_std
	score_one = SA_score_norm + logp_norm + cycle_score_norm
  
	return score_one

def shell(cmd, shell=False):

	if shell:
		p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	else:
		cmd = cmd.split()
		p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	output, err = p.communicate()
	return output

def write_xtb_input_file(fragment, fragment_name):
	number_of_atoms = fragment.GetNumAtoms()
	charge = Chem.GetFormalCharge(fragment)
	symbols = [a.GetSymbol() for a in fragment.GetAtoms()] 
	for i,conf in enumerate(fragment.GetConformers()):
		file_name = fragment_name+"+"+str(i)+".xyz"
		with open(file_name, "w") as file:
			file.write(str(number_of_atoms)+"\n")
			file.write("title\n")
			for atom,symbol in enumerate(symbols):
				p = conf.GetAtomPosition(atom)
				line = " ".join((symbol,str(p.x),str(p.y),str(p.z),"\n"))
				file.write(line)
			if charge !=0:  
				file.write("$set\n")
				file.write("chrg "+str(charge)+"\n")
				file.write("$end")

def get_structure(mol,n_confs):
	mol = Chem.AddHs(mol)
	new_mol = Chem.Mol(mol)

	AllChem.EmbedMultipleConfs(mol,numConfs=n_confs,useExpTorsionAnglePrefs=True,useBasicKnowledge=True,maxAttempts=5000)
	energies = AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters=2000, nonBondedThresh=100.0)

	energies_list = [e[1] for e in energies]
	min_e_index = energies_list.index(min(energies_list))

	new_mol.AddConformer(mol.GetConformer(min_e_index))

	return new_mol

def compute_absorbance(mol,n_confs,path):
	mol = get_structure(mol,n_confs)
	dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
	os.mkdir(dir)
	os.chdir(dir)
	write_xtb_input_file(mol, 'test')
	shell(path+'/xtb4stda test+0.xyz',shell=False)
	out = shell(path+'/stda_v1.6.1 -xtb -e 10',shell=False)
	#data = str(out).split('Rv(corr)\\n')[1].split('alpha')[0].split('\\n') # this gets all the lines
	data = str(out).split('Rv(corr)\\n')[1].split('(')[0]
	wavelength, osc_strength = float(data.split()[2]), float(data.split()[3])
	os.chdir('..')
	shutil.rmtree(dir)

	return wavelength, osc_strength

def absorbance_target(mol,args):
	n_confs, path, target, sigma, threshold = args
	try:
		wavelength, osc_strength = compute_absorbance(mol,n_confs,path)
	except:
		return 0.0
	
	score = GaussianModifier(wavelength, target, sigma) 
	score += ThresholdedLinearModifier(osc_strength,threshold)

	return score

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

def connect_amine_with_struc(amine, structure_smi='[H]c1c([H])c([C@]([H])([O-])[C@@]([H])(C(=O)OC([H])([H])[H])C([H])([H])[N@+]23C([H])([H])C([H])([H])[C@]([H])(C([H])([H])C2([H])[H])C([H])([H])C3([H])[H])c([H])c([H])c1[N+](=O)[O-]'):
	'''	Takes mol that contains amine and connects it to 5-sr, returns list of 6*n_amine products (n_amine = number of amine substructe in input mol)	'''
	struc = Chem.MolFromSmiles(structure_smi)
	connect_smarts = '[NX3;H0;D3;!+1]([*:1])([*:2])[*:3].[C$([*]([NX4;H0;D4&+1])([CH1])):4][NX4;H0;D4&+1:9]>>[NX4;H0;D4&+1]([*:1])([*:2])([*:3])[C:4]'
	rxn = AllChem.ReactionFromSmarts(connect_smarts)
	ps = rxn.RunReactants((amine,struc))
	return ps

def number_of_conformers(mol):
	n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
	n_confs = 5 + 5*n_rot
	return n_confs

def compute_energy_diff(amine, n_confs=None):
	'''	Takes mol containing amine and calculates its energy as well as the energy of amine+5-st molecule and returns energy difference 	'''	
	if n_confs is None:
		n_confs = number_of_conformers(amine)
	e_cat = compute_energy(amine,n_confs)
	products = connect_amine_with_struc(amine)
	unique = get_unique_amine_products(amine,products)
	energy_of_conformers = []
	for conf in unique:
		conf = cleanup(conf)
		energy_of_conformers.append(compute_energy(conf,n_confs))
	min_e_conf = min(energy_of_conformers)
	energy_diff = (-52.277827212938 + e_cat) - min_e_conf 
	return energy_diff

def energydiff2score(energies):
	score = [energie * -1 for energie in energies]
	return score

def cat_scoring(amine):
	energy_diff = compute_energy_diff(amine)
	score = energydiff2score(energy_diff)
	return score

def count_substruc_match(mol, pattern):
	'''	Counts how ofter a pattern occurs in mol, returns count	'''
	count = len(mol.GetSubstructMatches(pattern))
	return count

def get_unique_amine_products(amine, products):
	'''	Returns unique products from connect_amine_with_struc	'''
	tert_amine = Chem.MolFromSmarts('[NX3;H0;D3;!+1]')
	quart_amine = Chem.MolFromSmarts('[NX4;H0;D4&+1]')
	n = count_substruc_match(amine, tert_amine)
	unique = []
	for n in np.arange(n):
		unique.append(products[n*6][0])
	return unique

def cleanup(mol):
	'''	Returns nice 2D mol'''
	Chem.SanitizeMol(mol)
	smi = Chem.MolToSmiles(mol)
	new_mol = Chem.MolFromSmiles(smi)
	return new_mol
  
# GuacaMol article https://arxiv.org/abs/1811.09621
# adapted from https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/fingerprints.py

def get_ECFP4(mol):
	return AllChem.GetMorganFingerprint(mol, 2)

def get_ECFP6(mol):
	return AllChem.GetMorganFingerprint(mol, 3)

def get_FCFP4(mol):
	return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

def get_FCFP6(mol):
	return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)

def rediscovery(mol,args):
	target = args[0]
	try:
		fp_mol = get_ECFP4(mol)
		fp_target = get_ECFP4(target)

		score = TanimotoSimilarity(fp_mol, fp_target)

		return score
  
	except:
		print('Failed ',Chem.MolToSmiles(mol))
		return None

def MCS(mol,args):
	target = args[0]
	try:
		mcs = rdFMCS.FindMCS([mol, target], bondCompare=rdFMCS.BondCompare.CompareOrderExact,ringMatchesRingOnly=True,completeRingsOnly=True)
		score = mcs.numAtoms/target.GetNumAtoms()
		return score
	
	except:
		print('Failed ',Chem.MolToSmiles(mol))
		return None

def similarity(mol,target,threshold):
	score = rediscovery(mol,target)
	if score:
		return ThresholdedLinearModifier(score,threshold)
	else:
		return None

# adapted from https://github.com/BenevolentAI/guacamol/blob/master/guacamol/score_modifier.py

def ThresholdedLinearModifier(score,threshold):
  	return min(score,threshold)/threshold

def GaussianModifier(score, target, sigma):
	try:
		score = np.exp(-0.5 * np.power((score - target) / sigma, 2.))
	except:
		score = 0.0

	return score




if __name__ == "__main__":
	n_confs = 20
	xtb_path = '/home/jhjensen/stda'
	target = 200.
	sigma = 50.
	threshold = 0.3
	smiles = 'Cc1occn1' # Tsuda I
	mol = Chem.MolFromSmiles(smiles)

	wavelength, osc_strength = compute_absorbance(mol,n_confs,xtb_path)
	print(wavelength, osc_strength)

	score = absorbance_target(mol,[n_confs, xtb_path, target, sigma, threshold])
	print(score)

