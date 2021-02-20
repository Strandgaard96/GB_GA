# %%
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') 

import copy
import json
import string
import time
import subprocess
import os
import shutil
import sys
sys.path.append('/home/julius/soft/')
from xyz2mol.xyz2mol import read_xyz_file, xyz2AC, xyz2mol, get_AC


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
    for i in range(num_confs):
        try:
            conf = AllChem.ConstrainedEmbed(mol,template,useTethers=True,randomseed=randomseed,getForceField=GetFF)
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
    if AllChem.MMFFHasAllMoleculeParams(mol):
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=confId, ignoreInterfragInteractions=False, nonBondedThresh=nonBondedThresh)
        for atom in constrained_atoms:
            ff.MMFFAddPositionConstraint(atom, maxDispl, forceConstant)
    else:
        ff = AllChem.UFFGetMoleculeForceField(mol,confId=-1,ignoreInterfragInteractions=False)
        for atom in constrained_atoms:
            ff.UFFAddPositionConstraint(atom, maxDispl, forceConstant)
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



def write_xtb_input_file(fragment, file_name):
    number_of_atoms = fragment.GetNumAtoms()
    charge = Chem.GetFormalCharge(fragment)
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()] 
    
    conformers = fragment.GetConformers()
    if not conformers:
        raise Exception('Mol is not embedded.')
    if not conformers[0].Is3D:
        raise Exception('Mol is not 3D.')
    for i,conf in enumerate(fragment.GetConformers()):
        with open(file_name, "w") as file:
            file.write(str(number_of_atoms)+"\n")
            file.write(f'{file_name}\n')
            for atom,symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = " ".join((symbol,str(p.x),str(p.y),str(p.z),"\n"))
                file.write(line)
    # print(f'Writing .CHRG file with charge {charge} for {Chem.MolToSmiles(fragment)}')
    if charge !=0:
        with open(".CHRG", "w") as _file:
            _file.write(str(charge))
    return os.path.abspath(file_name)

import random

def xtb_optimize(mol, name='xtbmol', constrains=None, scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True):
    dest = os.path.join(scratchdir, 'tmp_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4)))
    try:
        os.mkdir(dest)
    except:
        pass
    os.chdir(dest)
    try:
        xyz_file = write_xtb_input_file(mol, 'mol.xyz')
    except:
        os.chdir('..')
        raise
    true_charge = Chem.GetFormalCharge(mol)
    if constrains:
        subprocess.Popen(f'xtb mol.xyz --opt tight --input {constrains} --alpb methanol --json > out.out' ,shell=True)    
    else:
        subprocess.Popen('xtb mol.xyz --opt tight --alpb methanol --json > out.out' ,shell=True)
    time.sleep(5)
    while not os.path.isfile('xtbopt.xyz'):
        time.sleep(5)
    out_file = os.path.abspath('xtbopt.xyz')
    with open('xtbout.json', 'r') as _file:
        out = json.load(_file)
        energy_au = out['total energy']
    os.chdir('..')
    atoms, charge, coordinates = read_xyz_file(out_file) 
    new_mol = xyz2mol(atoms, coordinates, true_charge) # takes charge as defined before optimization
    if remove_tmp:
        shutil.rmtree(dest)
    return new_mol, energy_au


def get_connection_id(mol):
    for atom in mol.GetAtomWithIdx(0).GetNeighbors():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
            connection_id = atom.GetIdx()
            break
    return connection_id

def make_reactant(mol, reactant_dummy, n_confs=5, randomseed=100, xtb_opt=False, scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True):
    """Connects Catalyst with Reactant via dummy atom, returns min(E) conformer of all n_conf * n_tertary_amines conformers"""
    nonBondedThresh=100
    energyCutOff=400
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
        try:
            possible_reactant_conformers = ConstrainedEmbedMultipleConfs(possible_reactant, reactant_dummy, n_confs, randomseed)
        except:
            print(f'Was not able to embed {Chem.MolToSmiles(mol)}, trying next conformer.')
            continue
        # constrained_atoms = possible_reactant.GetSubstructMatch(Chem.MolFromSmarts('[#6](/[#6](=[#6](/[#8]-[#6](-[H])(-[H])-[H])-[#8-])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H]).[H]-[#8]-[#6](-[H])(-[H])-[H]')) + possible_reactant.GetSubstructMatch(Chem.MolFromSmarts('[#6](-[#6@@](-[#6](-[#8]-[#6](-[H])(-[H])-[H])=[#8])(-[H])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H]).[#8-]-[#6](-[H])(-[H])-[H]'))
        constrained_atoms = []
        for atom in possible_reactant.GetAtoms():
            if atom.GetIdx() < get_connection_id(possible_reactant):
                constrained_atoms.append(atom.GetIdx())
            else:
                break  
        minEconf, minE = get_constr_minE_conf(possible_reactant_conformers, constrained_atoms, nonBondedThresh=nonBondedThresh)
        possible_reactants.append(minEconf)
        energies.append(minE)
    
    # if not possible_reactants:
    #     raise Exception('')

    # getting lowest energy reactant regioisomer
    reactant_minE = min(energies)
    min_e_index = energies.index(reactant_minE)
    reactant = possible_reactants[min_e_index]
    
    if xtb_opt:
        reactant, xtb_energy = xtb_optimize(reactant, scratchdir=scratchdir, remove_tmp=remove_tmp)
        reactant_minE = xtb_energy

    if reactant_minE > energyCutOff:
        print(f'Energy exceeded {energyCutOff} ({reactant_minE:.2f}) while trying to optimize the reactant molecule with Catalyst: {Chem.MolToSmiles(mol)}.')

    return reactant, reactant_minE

def make_product(mol, reactant, product_dummy, n_confs=5, nItsUnconstrained=100, randomseed=100, xtb_opt=False, scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True):
    """Creates same Regioisomer of Cat+Product_dummy and ensures that similar Rotamer as Reactant is obtained"""
    energyCutOff=400
    # test embed
    if test_embed(mol):
        raise Exception('Mol is already embeded.')

    # create all possible products
    product = None
    possible_products_2d = connect_cat_2d(product_dummy, mol)

    # making sure to ge the same regioisomer as reactant
    # for atom in reactant.GetAtomWithIdx(0).GetNeighbors():
    #     if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
    #         connection_id = atom.GetIdx()
    #         break
    connection_id = get_connection_id(reactant)
    # connection_id = reactant.GetSubstructMatch(Chem.MolFromSmarts('[#7+]-[#6](-[#6@@](-[#6](-[#8]-[#6](-[H])(-[H])-[H])=[#8])(-[H])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H])'))[0]
    for possible_product in possible_products_2d:
        test_connection_id = get_connection_id(possible_product) #possible_product.GetSubstructMatch(Chem.MolFromSmarts('[#7+]-[#6](/[#6](=[#6](/[#8]-[#6](-[H])(-[H])-[H])-[#8-])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H])'))[0]
        if int(connection_id) == int(test_connection_id):
            product = AllChem.ConstrainedEmbed(possible_product, product_dummy, useTethers=True, randomseed=randomseed)
    if product is None:
        raise Exception(f'No match between reactant and product regioisomers.')
    # test reasonability of embedding
    if AllChem.MMFFHasAllMoleculeParams(product):
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
    else:
        ff = AllChem.UFFGetMoleculeForceField(product,confId=-1,ignoreInterfragInteractions=False)
        ff.Initialize()
        pre_energy = ff.CalcEnergy()
        if pre_energy > energyCutOff:
            # try 5 times with different seed
            randomseeds = [x+randomseed for x in [1,2,3,4,5]]
            for seed in randomseeds:
                # embed again if it failed
                # print(f'Trying seed {seed}')
                product = AllChem.ConstrainedEmbed(possible_product, product_dummy, useTethers=True, randomseed=seed)
                ff = AllChem.UFFGetMoleculeForceField(product,confId=-1,ignoreInterfragInteractions=False)
                ff.Initialize()
                pre_energy = ff.CalcEnergy()
                if pre_energy < energyCutOff:
                    break


    # optimize product with constraints
    # constrain the core of the complex
    # constrained_atoms = product.GetSubstructMatch(Chem.MolFromSmarts('[#6](/[#6](=[#6](/[#8]-[#6](-[H])(-[H])-[H])-[#8-])-[#6@](-[H])(-[#6]1:[#6](:[#6](:[#6](:[#6](:[#6]:1-[H])-[H])-[#7+](=[#8])-[#8-])-[H])-[H])-[#8]-[H])(-[H])(-[H]).[H]-[#8]-[#6](-[H])(-[H])-[H]'))
    constrained_atoms = []
    for atom in product.GetAtoms():
        if atom.GetIdx() < get_connection_id(product):
            constrained_atoms.append(atom.GetIdx())
        else:
            break    
    
    if AllChem.MMFFHasAllMoleculeParams(product):
        for atom in constrained_atoms:
            ff.MMFFAddPositionConstraint(atom, 0, 1e3)
    else:
        for atom in constrained_atoms:
            ff.UFFAddPositionConstraint(atom, 0, 1e3)
            
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

    if xtb_opt:
            product, xtb_energy = xtb_optimize(product, scratchdir=scratchdir, remove_tmp=remove_tmp)
            product_energy = xtb_energy

    if product_energy > energyCutOff:
        print(f'Energy exceeded {energyCutOff} ({product_energy:.2f}) while trying to optimize the product molecule with Catalyst: {Chem.MolToSmiles(mol)}.')
    

    return product, product_energy
# %%
