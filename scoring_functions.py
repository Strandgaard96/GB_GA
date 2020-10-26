'''
Written by Jan H. Jensen 2018
'''

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

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
import time



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

def calculate_scores(population,function,scoring_args,n_cpus):
	scores = []
	for gene in population:
		score = function(gene,scoring_args)
		scores.append(score)

	return scores 

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
  

# if __name__ == "__main__":
# 	n_confs = 20
# 	xtb_path = '/home/jhjensen/stda'
# 	target = 200.
# 	sigma = 50.
# 	threshold = 0.3
# 	smiles = 'Cc1occn1' # Tsuda I
# 	mol = Chem.MolFromSmiles(smiles)

# 	wavelength, osc_strength = compute_absorbance(mol,n_confs,xtb_path)
# 	print(wavelength, osc_strength)

# 	score = absorbance_target(mol,[n_confs, xtb_path, target, sigma, threshold])
# 	print(score)

