from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from scoring_functions import shell, write_xtb_input_file

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') 

import numpy as np
import os
import shutil
import copy
import string
import random
import time
from time import sleep
import subprocess
from scoring_functions import shell

import sys
sys.path.append("../") # go to parent dir
from xyz2mol.xyz2mol import read_xyz_file, xyz2AC, xyz2mol, get_AC

# useless xtb sp
import numpy as np
from xtb.interface import (
    XTBException,
    Molecule,
    Calculator,
    Results,
    Param,
    Solvent,
)


def rdkit2xtb(mol):
    if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
        raise Exception('Implicit Hydrogens')
    # Get Atomic Numbers
    numbers = []
    for atom in molH.GetAtoms():
        numbers.append(atom.GetAtomicNum())
    numbers = np.array(numbers)
    # Get Atomic Positions
    all_positions = []
    conformers = mol.GetConformers()
    if not conformers:
        raise Exception('Mol is not embedded')
    else:
        for conformer in conformers:
            if not conformer.Is3D():
                raise Exception('Conformer is not 3D')
            positions = conformer.GetPositions()
            all_positions.append(positions)
    return numbers, all_positions

def minE_index(numbers, all_positions):
    energies = [ ]
    for i, positions in enumerate(all_positions):
        calc = Calculator(Param.GFN2xTB, numbers, positions)
        calc.set_solvent(Solvent.methanol)
        res = calc.singlepoint()
        energies.append(res.get_energy())
    return energies.index(min(energies)), min(energies)

def get_structure(start_mol,n_confs,randomSeed=-1, returnE=False):
    mol = Chem.AddHs(start_mol)
    new_mol = Chem.Mol(mol)

    confIDs = AllChem.EmbedMultipleConfs(mol,numConfs=n_confs,useExpTorsionAnglePrefs=True,useBasicKnowledge=True,maxAttempts=5000,randomSeed=randomSeed)
    energies = AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters=2000, nonBondedThresh=100.0)

    energies_list = [e[1] for e in energies]
    min_e_index = energies_list.index(min(energies_list))

    new_mol.AddConformer(mol.GetConformer(min_e_index))
    if returnE:
        return new_mol, min(energies_list)
    else:
        return new_mol


def get_xtb_energy(out):
    '''	Returns electronic energy calculated by xTB	'''
    if 'total energy' in str(out):
        xtb_energy = float(str(out).split('total energy')[1].split('Eh')[0])
    else:
        xtb_energy = 0
    return xtb_energy

def compute_energy(mol,n_confs,randomSeed=-1):
    '''	Generates n_confs conformers from input mol, starts xTB calculation with gbsa MeOH solvent model and returns electronic energy '''
    mol = get_structure(mol,n_confs,randomSeed=randomSeed)
    dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    os.mkdir(dir)
    os.chdir(dir)
    write_xtb_input_file(mol, 'test')
    out = shell('xtb test+0.xyz --opt --gbsa methanol', shell=False)
    energy = get_xtb_energy(out)
    os.chdir('..')
    shutil.rmtree(dir)
    return energy

# def connect_amine_with_struc(rdkit_mol, structure_smi='[H]c1c([H])c([C@]([H])([O-])[C@@]([H])(C(=O)OC([H])([H])[H])C([H])([H])[N@+]23C([H])([H])C([H])([H])[C@]([H])(C([H])([H])C2([H])[H])C([H])([H])C3([H])[H])c([H])c([H])c1[N+](=O)[O-]'):
#     '''	Takes mol that contains amine and connects it to 5-sr, returns list of 6*n_amine products (n_amine = number of amine substructe in input mol)	'''
#     struc = Chem.MolFromSmiles(structure_smi)
#     connect_smarts = '[#7X3;H0;D3;!+1]([*:1])([*:2])[*:3].[C$([*]([#7X4;H0;D4&+1])([CH1])):4][#7X4;H0;D4&+1]>>[#7X4;H0;D4&+1]([*:1])([*:2])([*:3])[C:4]'
#     rxn = AllChem.ReactionFromSmarts(connect_smarts)
#     ps = rxn.RunReactants((rdkit_mol,struc))
#     return ps

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

def test_embed(mol):
    conformers = mol.GetConformers()
    if not conformers:
        # print('Catalyst is not embedded.')
        return False
    elif not conformers[0].Is3D:
        # print('Conformer is not 3D.')
        return False
    else:
        return True

def ac_from_xyz(xyz_file):
    atoms, charge, coordinates = read_xyz_file(xyz_file)
    ac = xyz2AC(atoms, coordinates, charge)[0]
    return ac

def same_ac(ac1,ac2):
    return (ac1 == ac2).all()

def write_xtb_input_file(fragment, fragment_name):
    number_of_atoms = fragment.GetNumAtoms()
    charge = Chem.GetFormalCharge(fragment)
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()] 
    file_name = fragment_name +".xyz"
    conformers = fragment.GetConformers()
    if not conformers:
        raise Exception('Mol is not embedded.')
    if not conformers[0].Is3D:
        raise Exception('Mol is not 3D.')
    for i,conf in enumerate(fragment.GetConformers()):
        with open(file_name, "w") as file:
            file.write(str(number_of_atoms)+"\n")
            file.write(f'{fragment_name}\n')
            for atom,symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = " ".join((symbol,str(p.x),str(p.y),str(p.z),"\n"))
                file.write(line)
            if charge !=0:
                file.write("$set\n")
                file.write("chrg "+str(charge)+"\n")
                file.write("$end")
    return file_name

def slurm_xtb_path(reactant_xyz, product_xyz, k_push, k_pull, alp, inp_file='path.inp'):
    jobid = os.popen('submit_xtb_path_juls ' + reactant_xyz + ' ' + product_xyz + ' ' + str(k_push) + ' ' + str(k_pull) + ' ' + str(alp) + ' ' + str(inp_file)).read()
    jobid = int(jobid.split()[-1])
    return {jobid}

def wait_for_jobs_to_finish(job_ids):
    """
    This script checks with slurm if a specific set of jobids is finished with a
    frequency of 1 minute.
    Stops when the jobs are done.
    """
    while True:
        job_info1 = os.popen("squeue -p coms").readlines()[1:]
        job_info2 = os.popen("squeue -u julius").readlines()[1:]
        current_jobs1 = {int(job.split()[0]) for job in job_info1}
        current_jobs2 = {int(job.split()[0]) for job in job_info2}
        current_jobs = current_jobs1|current_jobs2
        if current_jobs.isdisjoint(job_ids):
            break
        else:
            time.sleep(10)

def get_xtbpath_energy(out_file):
    barriers = []
    rmse_prod_to_endpath = []
    reactions_completed = []
    with open(out_file, 'r') as _file:
        line = _file.readline()
        while not barriers:
            if "energies in kcal/mol, RMSD in Bohr" in line:
                for _ in range(3):
                    line = _file.readline()
                    data = line.split()
                    if data[0] == "WARNING:":
                        line = _file.readline()
                        data = line.split()
                    try:
                        barriers.append(np.float(data[3]))
                    except:
                        print(data)
                    rmsd_idx = 14
                    if data[4] == 'dE:*******':
                        rmsd_idx = 13
                    if 'dE:-' in data[4]:
                        rmsd_idx = 13
                    try:
                        rmsd = np.float(data[rmsd_idx])
                    except:
                        print(data)
                    rmse_prod_to_endpath.append(rmsd)
                    if rmsd < 0.3:
                        reactions_completed.append(True)
                    else:
                        reactions_completed.append(False)
            line = _file.readline()
    
    return barriers, rmse_prod_to_endpath

def xtb_optimize(mol, name='xtbmol', return_mol=False, check_AC=False, constrains=None):
    dest = 'tmp'
    try:
        os.mkdir(dest)
    except:
        pass
    os.chdir(dest)
    try:
        xyz_file = write_xtb_input_file(mol, 'mol')
    except:
        os.chdir('..')
        raise
    if constrains:
        subprocess.Popen(f'xtb mol.xyz --opt --input {constrains} --gbsa methanol' ,shell=True)    
    else:
        subprocess.Popen('xtb mol.xyz --opt --gbsa methanol' ,shell=True)
    sleep(5)
    while not os.path.isfile('xtbopt.xyz'):
        sleep(5)
    out_file = os.path.abspath('xtbopt.xyz')
    name = name + '.xyz'
    os.rename(out_file, f'../{name}')
    os.chdir('..')
    shutil.rmtree('tmp')
    if return_mol:
        atoms, charge, coordinates = read_xyz_file(name)
        new_mol = xyz2mol(atoms, coordinates, charge)
        if check_AC:
            return name, new_mol, same_ac(get_AC(mol), get_AC(new_mol))
        else:
            return new_mol
    else:
        if check_AC:
            return name, same_ac(get_AC(mol), ac_from_xyz(name))
        else:
            return name
        return name

def constrained_optimization(mol, maxIts=10000, maxDispl=0.1, forceConstant=1e5, confId=-1, ignoreIncomplete=False):
    constrained_atoms = mol.GetSubstructMatch(Chem.MolFromSmarts('[#6](/[#6](=[#6](/[#8]-[#6](-[H])(-[H])-[H])-[#8-])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H]).[H]-[#8]-[#6](-[H])(-[H])-[H]')) + mol.GetSubstructMatch(Chem.MolFromSmarts('[#6](-[#6@@](-[#6](-[#8]-[#6](-[H])(-[H])-[H])=[#8])(-[H])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H]).[#8-]-[#6](-[H])(-[H])-[H]'))
    if len(constrained_atoms) == 0:
        raise Exception('No match with core.')
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=confId, ignoreInterfragInteractions=False)
    for atom in constrained_atoms:
        ff.MMFFAddPositionConstraint(atom, maxDispl, forceConstant)
    ff.Initialize()
    out = ff.Minimize(maxIts=maxIts)
    if out == 1:
        if ignoreIncomplete:
            # print('Optimization incomplete')
            energy = ff.CalcEnergy()
            return energy
        else:
            raise Exception('Optimization failed.')
    else:
        energy = ff.CalcEnergy()
        return energy

def connect_cat_2d(mol_with_dummy, cat):
    dummy = Chem.MolFromSmarts('[99*]')
    mols = []
    cat = Chem.AddHs(cat)
    for amine in cat.GetSubstructMatches(Chem.MolFromSmarts('[#7X3;H0;D3;!+1]')):
        mol = AllChem.ReplaceSubstructs(mol_with_dummy, dummy, cat, replacementConnectionPoint=amine[0])[0]
        quart_amine = mol.GetSubstructMatch(Chem.MolFromSmarts('[#7X4;H0;D4;!+1]'))[0]
        mol.GetAtomWithIdx(quart_amine).SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        mol.RemoveAllConformers()
        mols.append(mol)
    return mols

def addConf2Mol(mol,conf):
    mol.AddConformer(conf.GetConformer(-1),assignId=True)
    return mol

def ConstrainedEmbedMultipleConfs(mol,template,num_confs,useTethers=True,randomseed=100,retainOnlyMinEConf=False):
    GetFF = lambda x,confId=-1:AllChem.MMFFGetMoleculeForceField(x,AllChem.MMFFGetMoleculeProperties(x),confId=confId, ignoreInterfragInteractions=False)
    new_mol = copy.deepcopy(mol)
    energies = []
    confs = []
    for i in range(num_confs):
        conf = AllChem.ConstrainedEmbed(mol,template,useTethers=True,randomseed=randomseed,getForceField=GetFF)
        if retainOnlyMinEConf:
            energies.append(constrained_optimization(conf))
            confs.append(conf)
        else:
            new_mol = addConf2Mol(new_mol,conf)
    if retainOnlyMinEConf:
        min_e_index = energies.index(min(energies))
        new_mol = confs[min_e_index]
        return new_mol, min(energies)
    else:
        return new_mol

reactant_dummy = None #Chem.SDMolSupplier('catalyst/structures/reactant_dummy.sdf',removeHs = False)[0]
product_dummy = None #Chem.SDMolSupplier('catalyst/structures/product_dummy.sdf',removeHs = False)[0]

def make_reactant(mol, reactant_dummy = reactant_dummy, energyCutOff=400, n_confs=5, retainOnlyMinEConf=True, xtb_opt=True):
    # test embed
    if test_embed(mol):
        raise Exception('Mol is already embeded.')
    
    ## REACTANT
    # get min(Energy) conformer of each possible reactant
    possible_reactants_2d = connect_cat_2d(reactant_dummy, mol)
    possible_reactants = []
    energies = []
    for possible_reactant in possible_reactants_2d:
        # create Multiple Conformations and get min(Energy) Conformer
        minEconf, minE = ConstrainedEmbedMultipleConfs(possible_reactant, reactant_dummy, n_confs, retainOnlyMinEConf=True)
        possible_reactants.append(minEconf)
        energies.append(minE)

    # getting lowest energy reactant regioisomer
    reactant_minE = min(energies)
    min_e_index = energies.index(reactant_minE)
    reactant = possible_reactants[min_e_index]

    if xtb_opt:
        _, opt1, ac_check1 = xtb_optimize(reactant, return_mol=True, check_AC=True, constrains='/home/julius/soft/GB-GA/constr_opt.inp')
        if ac_check1:
            _, opt2, ac_check2 = xtb_optimize(opt1, return_mol=True, check_AC=True, constrains='/home/julius/soft/GB-GA/constr_oh.inp')
            if ac_check2:
                reactant = opt2
            else:
                reactant = opt1
        else:
            print('xTB optimization changed connectivity. Proceed with MMFF optimized geometry.')

    
    if min(energies) > energyCutOff:
        print(f'Energy exceeded {energyCutOff} ({min(energies):.2f}) while trying to optimize the reactant molecule.')
        reactant = possible_reactants[min_e_index]

    return reactant, reactant_minE

def make_product(mol, reactant, product_dummy=product_dummy, energyCutOff=400, n_confs=5, nItsUnconstrained=100):
    # test embed
    if test_embed(mol):
        raise Exception('Mol is already embeded.')

    # create all possible products
    product = None
    possible_products_2d = connect_cat_2d(product_dummy, mol)

    # making sure to ge the same regioisomer as reactant
    connection_id = reactant.GetSubstructMatch(Chem.MolFromSmarts('[#7+]-[#6](-[#6@@](-[#6](-[#8]-[#6](-[H])(-[H])-[H])=[#8])(-[H])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H])'))[0]
    for possible_product in possible_products_2d:
        test_connection_id = possible_product.GetSubstructMatch(Chem.MolFromSmarts('[#7+]-[#6](/[#6](=[#6](/[#8]-[#6](-[H])(-[H])-[H])-[#8-])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H])'))[0]
        if int(connection_id) == int(test_connection_id):
            product = AllChem.ConstrainedEmbed(possible_product, product_dummy, useTethers=True, randomseed=100)
    if product is None:
        raise Exception('No match between reactant and product regioisomers.')
    
    # test reasonability of embedding
    mmff_props = AllChem.MMFFGetMoleculeProperties(product)
    ff = AllChem.MMFFGetMoleculeForceField(product, mmff_props, ignoreInterfragInteractions=False)
    ff.Initialize()
    pre_energy = ff.CalcEnergy()
    if pre_energy > energyCutOff:
        # try 5 times with different seed
        randomseeds = [101,102,103,104,105]
        for seed in randomseeds:
            # embed again if it failed
            print(f'Trying seed {seed}')
            product = AllChem.ConstrainedEmbed(possible_product, product_dummy, useTethers=True, randomseed=seed)
            mmff_props = AllChem.MMFFGetMoleculeProperties(product)
            ff = AllChem.MMFFGetMoleculeForceField(product, mmff_props, ignoreInterfragInteractions=False)
            ff.Initialize()
            pre_energy = ff.CalcEnergy()
            if pre_energy < energyCutOff:
                break     

    # optimize product with constraints
    # constrain the core of the complex
    constrained_atoms = product.GetSubstructMatch(Chem.MolFromSmarts('[#6](/[#6](=[#6](/[#8]-[#6](-[H])(-[H])-[H])-[#8-])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H]).[H]-[#8]-[#6](-[H])(-[H])-[H]'))
    for atom in constrained_atoms:
        ff.MMFFAddPositionConstraint(atom, 0, 1e2)
            
    # add constrains that pull atoms of the catalyst in the product on to the positions of the atoms of the catalyst in the reactant
    molHs = Chem.AddHs(mol)
    reactant_cat_atoms = reactant.GetSubstructMatch(molHs)
    product_cat_atoms = product.GetSubstructMatch(molHs)
    reacconf = reactant.GetConformer()
    for atom in product_cat_atoms:
        p = reacconf.GetAtomPosition(atom)
        pIdx = ff.AddExtraPoint(p.x,p.y,p.z,fixed=True)-1
        ff.AddDistanceConstraint(pIdx, atom,0,0,1e2)
    ff.Initialize()
    ff.Minimize(maxIts=1000000, energyTol=1e-4,forceTol=1e-3)

    # relax product geometry without additional catalyst constraint
    product_energy = constrained_optimization(product, maxIts=nItsUnconstrained, maxDispl=0.01, forceConstant=1e4, ignoreIncomplete=True)

    if product_energy > energyCutOff:
        print(f'Energy exceeded {energyCutOff} ({product_energy:.2f}) while trying to optimize the product molecule.')
    
    return product, product_energy


