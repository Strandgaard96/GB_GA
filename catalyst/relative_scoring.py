from rdkit import Chem
from rdkit.Chem import AllChem

from scoring_functions import shell

# from rdkit import RDLogger 
# RDLogger.DisableLog('rdApp.*') 

import copy
import json
import random
import string

# Make Molecules

def test_embed(mol):
    """Tests if mol is embeded, returns Boolean"""
    conformers = mol.GetConformers()
    if not conformers:
        # print('Catalyst is not embedded.')
        return False
    elif not conformers[0].Is3D:
        # print('Conformer is not 3D.')
        return False
    else:
        return True

def sdf2mol(sdf_file):
    mol = Chem.SDMolSupplier(sdf_file,removeHs = False)[0]
    return mol

def connect_cat_2d(mol_with_dummy, cat):
    """Replaces Dummy Atom [*] in Mol with Cat via tertiary Amine, return list of all possible regioisomers"""
    dummy = Chem.MolFromSmiles('*')
    mols = []
    cat = Chem.AddHs(cat)
    tert_amines = cat.GetSubstructMatches(Chem.MolFromSmarts('[#7X3;H0;D3;!+1]'))
    for amine in tert_amines:
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

def ConstrainedEmbedMultipleConfs(mol,template,num_confs,randomseed=100):
    """Embeds num_confs Confomers of mol while constraining core to template, returns mol with num_confs"""
    GetFF = lambda x,confId=-1:AllChem.MMFFGetMoleculeForceField(x,AllChem.MMFFGetMoleculeProperties(x),confId=confId, ignoreInterfragInteractions=False)
    new_mol = copy.deepcopy(mol)
    energies = []
    confs = []
    Hmol = Chem.AddHs(mol)
    for i in range(num_confs):
        try:
            conf = AllChem.ConstrainedEmbed(Hmol,template,useTethers=True,randomseed=randomseed,getForceField=GetFF)
            if 'conf' not in locals():
                print(f'Constrained Embedding failed for mol :{Chem.MolToSmiles(mol)}')
        except ValueError:
            print(f'Failed to embed {Chem.MolToSmiles(mol)}, trying again with different seed')
            conf = AllChem.ConstrainedEmbed(mol,template,useTethers=True,randomseed=randomseed+i,getForceField=GetFF)
        new_mol = addConf2Mol(new_mol,conf)
        randomseed += 1
    return new_mol

def get_constr_minE_conf(mol, constrained_atoms, nonBondedThresh=100):
    """Optimizes multiple conformers of mol and returns lowest energy one and its energy"""
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    conformers = mol.GetConformers()
    energies = []
    for conf in conformers:
        energies.append(constrained_optimization(mol, constrained_atoms, confId=conf.GetId(), nonBondedThresh=nonBondedThresh))
    min_e_index = energies.index(min(energies))
    new_mol.AddConformer(mol.GetConformer(min_e_index))
    return new_mol, min(energies)

def constrained_optimization(mol, constrained_atoms, maxIts=10000, maxDispl=0.1, forceConstant=1e3, confId=-1, ignoreIncomplete=False, nonBondedThresh=100):
    """Performs MMFF Optimization while fixing provided atoms"""
    if len(constrained_atoms) == 0:
        raise Exception('No match with core.')
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=confId, ignoreInterfragInteractions=False, nonBondedThresh=nonBondedThresh)
    for atom in constrained_atoms:
        ff.MMFFAddPositionConstraint(atom, maxDispl, forceConstant)
    ff.Initialize()
    out = ff.Minimize(maxIts=maxIts)
    if out == 1:
        if ignoreIncomplete:
            print('Optimization incomplete')
            energy = ff.CalcEnergy()
            return energy
        else:
            raise Exception('Optimization incomplete.')
    else:
        energy = ff.CalcEnergy()
        return energy

import string
import time
import subprocess
import os
import shutil
import sys
sys.path.append('../../soft/')
from xyz2mol.xyz2mol import read_xyz_file, xyz2AC, xyz2mol, get_AC

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

def xtb_optimize(mol, name='xtbmol', return_mol=False, check_AC=False, constrains=None):
    dest = 'tmp'+''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
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
    # if constrains:
    #     out = subprocess.Popen(f'xtb mol.xyz --opt --input {constrains} --gbsa methanol --json'.split() ,shell=False)    
    # else:
    #     out = subprocess.Popen('xtb mol.xyz --opt --gbsa methanol --json'.split() ,shell=False)
    if constrains:
        shell(f'xtb mol.xyz --opt tight --input {constrains} --gbsa methanol --json' ,shell=False)    
    else:
        shell('echo $HOME')
        shell('xtb mol.xyz --opt tight --gbsa methanol --json' ,shell=False)
    time.sleep(5)
    while not os.path.isfile('xtbout.json'):
        time.sleep(5)
    # time.sleep(1)
    out_file = os.path.abspath('xtbopt.xyz')
    name = name + '.xyz'
    os.rename(out_file, f'../{name}')
    with open('xtbout.json', 'r') as _file:
        out = json.load(_file)
        energy_au = out['total energy']
    os.chdir('..')
    shutil.rmtree(dest)
    if return_mol:
        atoms, charge, coordinates = read_xyz_file(name)
        new_mol = xyz2mol(atoms, coordinates, charge)
        if check_AC:
            return energy_au, new_mol, same_ac(get_AC(mol), get_AC(new_mol))
        else:
            return new_mol, energy_au
    else:
        if check_AC:
            return energy_au, same_ac(get_AC(mol), ac_from_xyz(name))
        else:
            return energy_au
        return energy_au

def make_reactant(mol, reactant_dummy, energyCutOff=400, n_confs=5, randomseed=100, xtb_opt=False, nonBondedThresh=100):
    """Connects Catalyst with Reactant via dummy atom, returns min(E) conformer of all n_conf * n_tertary_amines conformers"""
    # test embed
    if test_embed(mol):
        raise Exception('Mol is already embeded.')

    forcefield = 'UFF'
    if not is_parameterized(mol, forcefield=forcefield): # how do i handle initial molceules that can not be scored?
        print(f'help, SOS, Mol ({Chem.MolToSmiles(mol)}) is fucked up')
        # raise Exception(f'Mol ({Chem.MolToSmiles(mol)}) is not fully parameterized in {forcefield}.')
        return None, None
    else:
        print('it works?!')
        
    
    ## REACTANT
    # get min(Energy) conformer of each possible reactant   

    possible_reactants_2d = connect_cat_2d(reactant_dummy, mol)
    possible_reactants = []
    energies = []
    for possible_reactant in possible_reactants_2d:
        # create Multiple Conformations and get min(Energy) Conformer
        try:
            possible_reactant_conformers = ConstrainedEmbedMultipleConfs(possible_reactant, reactant_dummy, n_confs, randomseed)
        except ValueError:
            print(f'Was not able to embed {Chem.MolToSmiles(mol)}, trying next conformer.')
            continue
        constrained_atoms = possible_reactant.GetSubstructMatch(Chem.MolFromSmarts('[#6](/[#6](=[#6](/[#8]-[#6](-[H])(-[H])-[H])-[#8-])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H]).[H]-[#8]-[#6](-[H])(-[H])-[H]')) + possible_reactant.GetSubstructMatch(Chem.MolFromSmarts('[#6](-[#6@@](-[#6](-[#8]-[#6](-[H])(-[H])-[H])=[#8])(-[H])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H]).[#8-]-[#6](-[H])(-[H])-[H]'))
        minEconf, minE = get_constr_minE_conf(possible_reactant_conformers, constrained_atoms, nonBondedThresh=nonBondedThresh)
        possible_reactants.append(minEconf)
        energies.append(minE)
    
    # if not possible_reactants:
    #     raise Exception('')

    # getting lowest energy reactant regioisomer
    if not energies:
        print(f'something was wrong while doing {Chem.MolToSmiles(mol)}')
    reactant_minE = min(energies)
    min_e_index = energies.index(reactant_minE)
    reactant = possible_reactants[min_e_index]
    
    if min(energies) > energyCutOff:
        print(f'Energy exceeded {energyCutOff} ({min(energies):.2f}) while trying to optimize the reactant molecule.')
        reactant = possible_reactants[min_e_index]

    if xtb_opt:
        reactnat, xtb_energy = xtb_optimize(reactant, return_mol=True) ###########################################################!!!!!!!!!!!!!!!!!
        reactant_minE = xtb_energy

    return reactant, reactant_minE

from crossover import is_parameterized

def make_product(mol, reactant, product_dummy, energyCutOff=400, n_confs=5, nItsUnconstrained=100, randomseed=100, xtb_opt=False):
    """Creates same Regioisomer of Cat+Product_dummy and ensures that similar Rotamer as Reactant is obtained"""
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
            product = AllChem.ConstrainedEmbed(possible_product, product_dummy, useTethers=True, randomseed=randomseed)
    if product is None:
        raise Exception('No match between reactant and product regioisomers.')
    
    # test reasonability of embedding
    mmff_props = AllChem.MMFFGetMoleculeProperties(product)
    ff = AllChem.MMFFGetMoleculeForceField(product, mmff_props, ignoreInterfragInteractions=False)
    ff.Initialize()
    pre_energy = ff.CalcEnergy()
    if pre_energy > energyCutOff:
        # try 5 times with different seed
        randomseeds = [x+randomseed for x in [1,2,3,4,5]]
        for seed in randomseeds:
            # embed again if it failed
            # print(f'Trying seed {seed}')
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
        ff.MMFFAddPositionConstraint(atom, 0, 1e3)
            
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
    product_energy = constrained_optimization(product, constrained_atoms, maxIts=nItsUnconstrained, maxDispl=0.01, forceConstant=1e3, ignoreIncomplete=True)

    if product_energy > energyCutOff:
        print(f'Energy exceeded {energyCutOff} ({product_energy:.2f}) while trying to optimize the product molecule.')
    
    if xtb_opt:
            product, xtb_energy = xtb_optimize(product, return_mol=True)
            product_energy = xtb_energy

    return product, product_energy

reactant_dummy = sdf2mol('/home/julius/soft/GB-GA/catalyst/structures/ts7/rr_dummy.sdf')


from catalyst.cat import get_structure

def relative_scoring(mol, n_confs, xtb_opt=True):
    if xtb_opt:
        cat = get_structure(mol, n_confs, returnE=False)
        cat_energy = xtb_optimize(cat, return_mol=False)
    else:
        cat, cat_energy = get_structure(mol, n_confs, returnE=True)
    reactant, reactant_energy = make_reactant(mol, reactant_dummy, xtb_opt=xtb_opt)
    # product, product_energy = make_product(mol, reactant, product_dummy, nItsUnconstrained=0, xtb_opt=True)
    return -(reactant_energy-cat_energy)