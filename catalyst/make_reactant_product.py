# %%
from catalyst.utils import mols_from_smi_file, draw3d, sdf2mol, vis_trajectory, Timer
from xyz2mol.xyz2mol import read_xyz_file, xyz2AC, xyz2mol, get_AC
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')

import random
import copy
import json
import string
import time
import subprocess
import os
import shutil
import sys
sys.path.append('/home/julius/soft/')

# %%

def get_energy_from_xbt_xyz(xyz_file):
    with open(xyz_file, 'r') as _file:
        for i, line in enumerate(_file):
            if i == 1:
                energy = float(line.split(' ')[2])
                break
    return energy

def write_xtb_input_files(fragment, name, destination='.'):
    number_of_atoms = fragment.GetNumAtoms()
    charge = Chem.GetFormalCharge(fragment)
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
    conformers = fragment.GetConformers()
    file_paths = []
    for i, conf in enumerate(fragment.GetConformers()):
        conf_path = os.path.join(destination, f'conf{i:03d}')
        os.mkdir(conf_path)
        file_name = f'{name}{i:03d}.xyz'
        file_path = os.path.join(conf_path, file_name)
        with open(file_path, 'w') as _file:
            _file.write(str(number_of_atoms)+'\n')
            _file.write(f'{file_name}\n')
            for atom, symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = ' '.join((symbol, str(p.x), str(p.y), str(p.z), '\n'))
                _file.write(line)
        file_paths.append(file_path)
    if charge != 0:
        with open('.CHRG', 'w') as _file:
            _file.write(str(charge))
    return file_paths


def xtb_optimize(mol, name=None, constrains=None, method='gfn2', solvent='alpb methanol', opt_level='tight', scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True):
    if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
        raise Exception('Implicit Hydrogens')
    conformers = mol.GetConformers()
    if not conformers:
        raise Exception('Mol is not embedded')
    elif not conformers[-1].Is3D():
        raise Exception('Conformer is not 3D')
    org_dir = os.getcwd()
    true_charge = Chem.GetFormalCharge(mol)
    if not name:
        name = 'tmp_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    dest = os.path.join(
        os.path.abspath(scratchdir), name)
    os.mkdir(dest)
    energies = []
    out_files = []
    xyz_files = write_xtb_input_files(mol, 'xtbmol', destination=dest)
    for i, xyz_file in enumerate(xyz_files):
        conf_path = os.path.join(dest, f'conf{i:03d}')
        os.chdir(conf_path)
        if constrains:
            constrains_input = f'--input {constrains}'
        else:
            constrains_input = ''
        if solvent:
            solvent_input = f'--{solvent}'
        else:
            solvent_input = ''
        p = subprocess.Popen(
            f'xtb --{method} {xyz_file} --opt {opt_level} {constrains_input} {solvent_input} --json > out.out', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        out_file = 'xtbopt.xyz'
        try:
            energy = get_energy_from_xbt_xyz(out_file)
        except:
            energy = 1000
        # out_file = os.path.abspath('xtbopt.xyz')
        # with open('xtbout.json', 'r') as _file:
        #     out = json.load(_file)
        #     energy = out['total energy']
        energies.append(energy)
        out_files.append(os.path.abspath(out_file))
    os.chdir(org_dir)
    min_e_index = energies.index(min(energies))
    min_e_file = out_files[min_e_index]
    shutil.copy(min_e_file, os.path.join(dest, 'min_e_conf.xyz'))
    atoms, charge, coordinates = read_xyz_file(min_e_file)
    # takes charge as defined before optimization
    new_mol = xyz2mol(atoms, coordinates, true_charge)
    if remove_tmp:
        shutil.rmtree(dest)
    return new_mol, energy


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


def get_structure(start_mol, n_confs, randomSeed=-1, returnE=False):
    # mol = Chem.AddHs(start_mol)
    new_mol = Chem.Mol(mol)

    confIDs = AllChem.EmbedMultipleConfs(
        mol, numConfs=n_confs, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, maxAttempts=5000, randomSeed=randomSeed)
    energies = AllChem.MMFFOptimizeMoleculeConfs(
        mol, maxIters=2000, nonBondedThresh=100.0)

    energies_list = [e[1] for e in energies]
    min_e_index = energies_list.index(min(energies_list))

    new_mol.AddConformer(mol.GetConformer(min_e_index))
    if returnE:
        return new_mol, min(energies_list)
    else:
        return new_mol


def connect_cat_2d(mol_with_dummy, cat):
    """Replaces Dummy Atom [*] in Mol with Cat via tertiary Amine, return list of all possible regioisomers"""
    dummy = Chem.MolFromSmiles('*')
    mols = []
    cat = Chem.AddHs(cat)
    tert_amines = cat.GetSubstructMatches(
        Chem.MolFromSmarts('[#7X3;H0;D3;!+1]'))
    for amine in tert_amines:
        mol = AllChem.ReplaceSubstructs(
            mol_with_dummy, dummy, cat, replacementConnectionPoint=amine[0])[0]
        quart_amine = mol.GetSubstructMatch(
            Chem.MolFromSmarts('[#7X4;H0;D4;!+1]'))[0]
        mol.GetAtomWithIdx(quart_amine).SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        mol.RemoveAllConformers()
        mols.append(mol)
    return mols


def addMolAsConf(mol, mol2add):
    mol.AddConformer(mol2add.GetConformer(-1), assignId=True)
    return mol


def get_minE_conf(mol, constrained_atoms, nonBondedThresh=100, xtb_opt=True, method='gfn2'):
    """Optimizes multiple conformers of mol and returns lowest energy one and its energy"""
    if xtb_opt:
        new_mol, energy = xtb_optimize(
            mol, constrains='/home/julius/soft/GB-GA/catalyst/constr_opt.inp', remove_tmp=False, method=method)
        return new_mol, energy
    else:
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        conformers = mol.GetConformers()
        energies = []
        for conf in conformers:
            energy = constrained_optimization(
                mol, constrained_atoms, confId=conf.GetId(), nonBondedThresh=nonBondedThresh)
            energies.append(energy)
        min_e_index = energies.index(min(energies))
        new_mol.AddConformer(mol.GetConformer(min_e_index))
        return new_mol, min(energies)


def constrained_optimization(mol, constrained_atoms, maxIts=10000, maxDispl=0.1, forceConstant=1e3, confId=-1, ignoreIncomplete=False, nonBondedThresh=100):
    """Performs MMFF Optimization while fixing provided atoms"""
    if len(constrained_atoms) == 0:
        raise Exception('No match with core.')
    if AllChem.MMFFHasAllMoleculeParams(mol):
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=confId, ignoreInterfragInteractions=False, nonBondedThresh=nonBondedThresh)
        for atom in constrained_atoms:
            ff.MMFFAddPositionConstraint(atom, maxDispl, forceConstant)
    else:
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=confId, ignoreInterfragInteractions=False)
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


def get_connection_id(mol):
    for atom in mol.GetAtomWithIdx(0).GetNeighbors():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
            connection_id = atom.GetIdx()
            break
    return connection_id


def ConstrainedEmbed(mol, core, useTethers=True, coreConfId=-1, randomseed=2342, getForceField=AllChem.UFFGetMoleculeForceField, **kwargs):
    #
    #  Copyright (C) 2006-2017  greg Landrum and Rational Discovery LLC
    #
    #   @@ All Rights Reserved @@
    #  This file is part of the RDKit.
    #  The contents are covered by the terms of the BSD license
    #  which is included in the file license.txt, found at the root
    #  of the RDKit source tree.
    #
    force_constant = 1000.
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    if "." in Chem.MolToSmiles(mol):
        ci = AllChem.EmbedMolecule(mol, randomSeed=randomseed, **kwargs)  # jhj
    else:
        ci = AllChem.EmbedMolecule(
            mol, coordMap=coordMap, randomSeed=randomseed, **kwargs)

    if ci < 0:
        raise ValueError('Could not embed molecule.')

    algMap = [(j, i) for i, j in enumerate(match)]

    if not useTethers:
        # clean up the conformation
        ff = AllChem.UFFGetMoleculeForceField(mol)
        for i, idxI in enumerate(match):
            for j in range(i + 1, len(match)):
                idxJ = match[j]
                d = coordMap[idxI].Distance(coordMap[idxJ])
                ff.AddDistanceConstraint(idxI, idxJ, d, d, force_constant)
        ff.Initialize()
        n = 4
        more = ff.Minimize()
        while more and n:
            more = ff.Minimize()
            n -= 1
        # rotate the embedded conformation onto the core:
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)
        ff = AllChem.UFFGetMoleculeForceField(mol)
        conf = core.GetConformer()
        for i in range(core.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            q = mol.GetConformer().GetAtomPosition(i)
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            ff.AddDistanceConstraint(pIdx, match[i], 0, 0, force_constant)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    mol.SetProp('EmbedRMS', str(rms))
    return mol


def ConstrainedEmbedMultipleConfs(mol, template, num_confs, randomseed=100, lower_rms_cutoff=0, upper_rms_cutoff=200, max_tries=10):
    """Embeds num_confs Confomers of mol while constraining core to template, returns mol with num_confs"""
    GetFF = lambda x, confId=-1: AllChem.MMFFGetMoleculeForceField(
        x, AllChem.MMFFGetMoleculeProperties(x), confId=confId, ignoreInterfragInteractions=False)
    new_mol = copy.deepcopy(mol)
    energies = []
    confs = []
    n = 0
    while new_mol.GetNumConformers() < num_confs and n < max_tries:
        conf = ConstrainedEmbed(
            mol, template, useTethers=True, randomseed=randomseed, getForceField=GetFF)
        confs = new_mol.GetNumConformers()
        rmsds = []
        for j in range(confs):
            rmsds.append(AllChem.GetBestRMS(new_mol, conf, prbId=j))
        if all(item > lower_rms_cutoff and item < upper_rms_cutoff for item in rmsds):
            new_mol = addMolAsConf(new_mol, conf)
        randomseed += 1
        n += 1
    return new_mol


def make_reactant(mol, reactant_dummy, n_confs=5, randomseed=100, xtb_opt=False, scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True, preopt_method='gfn2'):
    """Connects Catalyst with Reactant via dummy atom, returns min(E) conformer of all n_conf * n_tertary_amines conformers"""
    t_make_reactant = Timer()
    t_make_reactant.start()
    nonBondedThresh = 100
    energyCutOff = 400
    # test embed
    if test_embed(mol):
        raise Exception('Mol is already embeded.')

    # REACTANT
    # get min(Energy) conformer of each possible reactant

    possible_reactants_2d = connect_cat_2d(reactant_dummy, mol)
    possible_reactants = []
    energies = []
    for possible_reactant in possible_reactants_2d:
        # create Multiple Conformations and get min(Energy) Conformer
        possible_reactant_conformers = ConstrainedEmbedMultipleConfs(
            possible_reactant, reactant_dummy, n_confs, randomseed)
        constrained_atoms = []  # only needed for FF get_minE
        for atom in possible_reactant.GetAtoms():
            if atom.GetIdx() < get_connection_id(possible_reactant):
                constrained_atoms.append(atom.GetIdx())
            else:
                break
        minEconf, minE = get_minE_conf(
            possible_reactant_conformers, constrained_atoms, nonBondedThresh=nonBondedThresh, method=preopt_method)
        possible_reactants.append(minEconf)
        energies.append(minE)

    # if not possible_reactants:
    #     raise Exception('')

    # getting lowest energy reactant regioisomer
    reactant_minE = min(energies)
    min_e_index = energies.index(reactant_minE)
    reactant = possible_reactants[min_e_index]

    if xtb_opt:
        t_xtb_opt = Timer()
        t_xtb_opt.start()
        reactant, xtb_energy = xtb_optimize(
            reactant, scratchdir=scratchdir, remove_tmp=remove_tmp, method=preopt_method)
        reactant_minE = xtb_energy
        t_xtb_opt.stop()

    if reactant_minE > energyCutOff:
        print(
            f'Energy exceeded {energyCutOff} ({reactant_minE:.2f}) while trying to optimize the reactant molecule with Catalyst: {Chem.MolToSmiles(mol)}.')

    t_make_reactant.stop()
    return reactant, reactant_minE


# from https://pschmidtke.github.io/blog/rdkit/3d-editor/2021/01/23/grafting-fragments.html
def getAttachmentVector(mol):
    """ for a fragment to add, search for the position of the attachment point and extract the atom id's of the attachment point and the connected atom (currently only single bond supported)
    mol: fragment passed as rdkit molecule
    return: tuple (atom indices)
    """

    rindex = -1
    rindexNeighbor = -1
    for atom in mol.GetAtoms():
        if(atom.GetAtomicNum() == 0):
            rindex = atom.GetIdx()
            neighbours = atom.GetNeighbors()
            if(len(neighbours) == 1):
                rindexNeighbor = neighbours[0].GetIdx()
            else:
                print("two attachment points not supported yet")
                return None
    return((rindex, rindexNeighbor))


def connectMols(mol1, mol2, atom1, atom2):
    """function copied from here https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py"""
    combined = Chem.CombineMols(mol1, mol2)
    emol = Chem.EditableMol(combined)
    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
    atom1_idx = atom1.GetIdx()
    atom2_idx = atom2.GetIdx()
    bond_order = atom2.GetBonds()[0].GetBondType()
    emol.AddBond(neighbor1_idx, neighbor2_idx +
                 mol1.GetNumAtoms(), order=bond_order)
    emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
    emol.RemoveAtom(atom1_idx)
    mol = emol.GetMol()
    return mol


def make_product(mol, reactant, product_dummy, n_confs=5, nItsUnconstrained=100, randomseed=100, xtb_opt=False, scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True, preopt_method='gfn2'):
    """Creates same Regioisomer of Cat+Product_dummy and ensures that similar Rotamer as Reactant is obtained"""
    energyCutOff = 400
    # test embed
    if test_embed(mol):
        raise Exception('Mol is already embeded.')

    # cut cat from reactant and attach to product_dummy
    bs = [reactant.GetBondBetweenAtoms(
        0, get_connection_id(reactant)).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(
        reactant, bs, addDummies=True, dummyLabels=[(1, 1)])
    fragments = AllChem.GetMolFrags(fragments_mol, asMols=True)

    for fragment in fragments:
        if fragment.HasSubstructMatch(mol):
            catalyst = fragment
            break
    cat_dummyidx = getAttachmentVector(catalyst)  # R N
    product_dummyidx = getAttachmentVector(product_dummy)  # R C

    rms = AllChem.AlignMol(product_dummy, catalyst, atomMap=(
        (product_dummyidx[0], cat_dummyidx[1]), (product_dummyidx[1], cat_dummyidx[0])))
    product = connectMols(product_dummy, catalyst, product_dummy.GetAtomWithIdx(
        product_dummyidx[0]), catalyst.GetAtomWithIdx(cat_dummyidx[0]))
    flags = Chem.SanitizeMol(product)

    # draw atoms of cat onto positions as in reactant
    constrained_atoms = []
    for atom in product.GetAtoms():
        if atom.GetIdx() < get_connection_id(product):
            constrained_atoms.append(atom.GetIdx())
        else:
            break

    AllChem.AlignMol(reactant, product, atomMap=list(
        zip(constrained_atoms, constrained_atoms)))  # atom 11 is included but whatever

    if AllChem.MMFFHasAllMoleculeParams(product):
        mmff_props = AllChem.MMFFGetMoleculeProperties(product)
        ff = AllChem.MMFFGetMoleculeForceField(
            product, mmff_props, ignoreInterfragInteractions=False)
        ff.Initialize()
        for atom in constrained_atoms:
            ff.MMFFAddPositionConstraint(atom, 0, 1e3)
    else:
        ff = AllChem.UFFGetMoleculeForceField(
            product, confId=-1, ignoreInterfragInteractions=False)
        ff.Initialize()
        for atom in constrained_atoms:
            ff.UFFAddPositionConstraint(atom, 0, 1e3)

    # add constrains that pull atoms of the catalyst in the product on to the positions of the atoms of the catalyst in the reactant
    molHs = Chem.AddHs(mol)
    reactant_cat_atoms = reactant.GetSubstructMatch(molHs)
    product_cat_atoms = product.GetSubstructMatch(molHs)
    reacconf = reactant.GetConformer()
    for atom in product_cat_atoms:
        p = reacconf.GetAtomPosition(atom)
        pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True)-1
        ff.AddDistanceConstraint(pIdx, atom, 0, 0, 1e10)
    ff.Initialize()
    ff.Minimize(maxIts=1000000, energyTol=1e-4, forceTol=1e-3)

    # relax product geometry without additional catalyst constraint
    product_energy = constrained_optimization(
        product, constrained_atoms, maxIts=nItsUnconstrained, maxDispl=0.01, forceConstant=1e3, ignoreIncomplete=True)

    if xtb_opt:
        product, xtb_energy = xtb_optimize(
            product, scratchdir=scratchdir, remove_tmp=remove_tmp, method=preopt_method)
        product_energy = xtb_energy

    if product_energy > energyCutOff:
        print(
            f'Energy exceeded {energyCutOff} ({product_energy:.2f}) while trying to optimize the product molecule with Catalyst: {Chem.MolToSmiles(mol)}.')

    return product, product_energy


# %%
if __name__ == '__main__':
    mols = mols_from_smi_file('/home/julius/thesis/data/QD_cats.smi')
    mol = mols[3]
    reactant_dummy = sdf2mol(
        '/home/julius/soft/GB-GA/catalyst/structures/reactant_dummy.sdf')
    product_dummy = sdf2mol(
        '/home/julius/soft/GB-GA/catalyst/structures/product_dummy.sdf')
    # %%
    reactant, reactant_energy = make_reactant(mol, reactant_dummy, xtb_opt=True)
    product, product_energy = make_product(mol, reactant, product_dummy, xtb_opt=True)
    # %%
    mols = mols_from_smi_file('/home/julius/thesis/data/QD_cats.smi')
    ts_energies = []
    for i, mol in enumerate(mols):
        moldir = f'/home/julius/thesis/scratch/1443976/mol00{i}'
        ts_energy = get_energy_from_path_ts(os.path.join(moldir, 'xtbpath_ts.xyz'))
        ts_energies.append(ts_energy)
    # draw3d([f'/home/julius/thesis/scratch/1443976/mol00{i}/xtbpath_ts.xyz'])
    # %%
