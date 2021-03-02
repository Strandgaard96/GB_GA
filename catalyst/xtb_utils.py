from rdkit import Chem
from rdkit.Chem import AllChem
from xyz2mol.xyz2mol import read_xyz_file, xyz2mol

import os
import random
import shutil
import string
import subprocess
import logging


# %%
def get_energy_from_xtb_sp(sp_file):
    with open(sp_file, 'r') as _file:
        for i, line in enumerate(_file):
            if i == 1:
                energy = float(line.split(' ')[2])
                break
    return energy

def write_xtb_input_files(fragment, name, destination='.'):
    number_of_atoms = fragment.GetNumAtoms()
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
            _file.write(f'{Chem.MolToSmiles(fragment)}\n')
            for atom, symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = ' '.join((symbol, str(p.x), str(p.y), str(p.z), '\n'))
                _file.write(line)
        file_paths.append(file_path)
    return file_paths

def xtb_optimize(mol, name=None, constrains=None, charge=None, method='gfn2', solvent='alpb methanol', opt_level='tight', scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True, return_file=False, numThreads=1, warning_logger=None):
    org_dir = os.getcwd()
    if isinstance(mol, Chem.rdchem.Mol):
        if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
            raise Exception('Implicit Hydrogens')
        conformers = mol.GetConformers()
        if not conformers:
            raise Exception('Mol is not embedded')
        elif not conformers[-1].Is3D():
            raise Exception('Conformer is not 3D')
        true_charge = Chem.GetFormalCharge(mol)
        if not name:
            name = 'tmp_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        dest = os.path.join(
            os.path.abspath(scratchdir), name)
        os.makedirs(dest)
        xyz_files = write_xtb_input_files(mol, 'xtbmol', destination=dest)
    else:
        xyz_files = [mol]
        dest = os.path.dirname(mol)
    energies = []
    out_files = []
    for i, xyz_file in enumerate(xyz_files):
        if isinstance(mol, Chem.rdchem.Mol):
            conf_path = os.path.join(dest, f'conf{i:03d}')
            os.chdir(conf_path)
        else:
            os.chdir(dest)
        if constrains:
            constrains_input = f'--input {constrains}'
        else:
            constrains_input = ''
        if solvent:
            solvent_input = f'--{solvent}'
        else:
            solvent_input = ''
        if charge:
            charge_input = f'--chrg {charge}'
        else:
            charge_input = ''
        os.environ['OMP_NUM_THREADS'] = f'{numThreads},1'
        os.environ['MKL_NUM_THREADS'] = f'{numThreads}'
        os.environ['OMP_STACKSIZE'] = '4G'
        p = subprocess.Popen(
            f'/home/julius/soft/xtb-6.3.3/bin/xtb --{method} {xyz_file} --opt {opt_level} {constrains_input} {solvent_input} {charge_input} --json > out.out', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        out_file = 'xtbopt.xyz'
        if not os.path.exists(out_file) and os.path.exists('xtblast.xyz'):
            if warning_logger:
                warning_logger.warning(f"Optimization for {Chem.MolToSmiles(mol)} did not converge: {os.path.dirname('xtblast.xyz')}")
            else:
                print(f"Optimization for {Chem.MolToSmiles(mol)} did not converge: {os.path.dirname('xtblast.xyz')}")
            out_file = 'xtblast.xyz'            
        try:
            energy = get_energy_from_xtb_sp(out_file)
        except:
            energy = 1000
        energies.append(energy)
        out_files.append(os.path.abspath(out_file))
    os.chdir(org_dir)
    min_e_index = energies.index(min(energies))
    min_e_file = out_files[min_e_index]
    shutil.copy(min_e_file, os.path.join(dest, 'min_e_conf.xyz'))
    energy = get_energy_from_xtb_sp(os.path.join(dest, 'min_e_conf.xyz'))
    if remove_tmp:
        shutil.rmtree(dest)
    if return_file:
        return os.path.abspath(min_e_file), energy
    else:
        atoms, _, coordinates = read_xyz_file(min_e_file)
        # takes charge as defined before optimization
        new_mol = xyz2mol(atoms, coordinates, true_charge)
        return new_mol, energy

def run_xtb_path(reactant_file, product_file, inp_file='/home/julius/thesis/data/path_template_allatoms.inp', charge=0, numThreads=1):
    os.environ['OMP_NUM_THREADS'] = f'{numThreads},1'
    os.environ['MKL_NUM_THREADS'] = f'{numThreads}'
    os.environ['OMP_STACKSIZE'] = '6G'
    path_dir = os.path.dirname(reactant_file)
    p = subprocess.Popen(f'/home/julius/soft/xtb-6.3.3/bin/xtb {reactant_file} --path {product_file} --input {inp_file} --gfn2 --chrg {charge} --alpb methanol --verbose > xtb_path.out', shell=True, cwd=path_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    ts_energy = get_energy_from_xtb_sp(os.path.join(path_dir, 'xtbpath_ts.xyz'))
    return ts_energy


#         def xtb_optimize_conformers(args):
#     xyz_file, i, dest, constrains, solvent, numThreads_per_conf, method, opt_level = args
#     org_dir = os.getcwd()
#     conf_path = os.path.join(dest, f'conf{i:03d}')
#     print(conf_path)
#     os.chdir(conf_path)
#     print(os.getcwd())
#     if constrains:
#         constrains_input = f'--input {constrains}'
#     else:
#         constrains_input = ''
#     if solvent:
#         solvent_input = f'--{solvent}'
#     else:
#         solvent_input = ''
#     os.environ['OMP_NUM_THREADS'] = f'{numThreads_per_conf},1'
#     os.environ['MKL_NUM_THREADS'] = f'{numThreads_per_conf}'
#     os.environ['OMP_STACKSIZE'] = '4G'
#     p = subprocess.Popen(
#         f'/home/julius/soft/xtb-6.3.3/bin/xtb --{method} {xyz_file} --opt {opt_level} {constrains_input} {solvent_input} --json > out.out', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     output, err = p.communicate()
#     out_file = 'xtbopt.xyz'
#     try:
#         energy = get_energy_from_xtb_sp(out_file)
#     except:
#         energy = 1000
#     os.chdir(org_dir)
#     return energy, os.path.abspath(out_file)

# def xtb_optimize(mol, name=None, constrains=None, method='gfn2', solvent='alpb methanol', opt_level='tight', scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True, numThreads=1):
#     if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
#         raise Exception('Implicit Hydrogens')
#     conformers = mol.GetConformers()
#     if not conformers:
#         raise Exception('Mol is not embedded')
#     elif not conformers[-1].Is3D():
#         raise Exception('Conformer is not 3D')
#     true_charge = Chem.GetFormalCharge(mol)
#     if not name:
#         name = 'tmp_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
#     dest = os.path.join(
#         os.path.abspath(scratchdir), name)
#     os.makedirs(dest)
#     energies = []
#     xyz_files = write_xtb_input_files(mol, 'xtbmol', destination=dest)
#     numThreads_per_conf = int(numThreads/len(xyz_files))
#     args_list = []
#     for i, xyz_file in enumerate(xyz_files):
#         args_list.append([xyz_file, i, dest, constrains, solvent, numThreads_per_conf, method, opt_level])
#     print(args_list)
#     with Pool(numThreads) as pool:
#         output = pool.map(xtb_optimize_conformers, args_list)
#     print(output)
#     out = np.array(output)
#     energies = list(out[:,0])
#     out_files = list(out[:,1])
#     print(energies)
#     print(out_files)
#     min_e_index = energies.index(min(energies))
#     min_e_file = out_files[min_e_index]
#     shutil.copy(min_e_file, os.path.join(dest, 'min_e_conf.xyz'))
#     atoms, charge, coordinates = read_xyz_file(min_e_file)
#     # takes charge as defined before optimization
#     new_mol = xyz2mol(atoms, coordinates, true_charge)
#     if remove_tmp:
#         shutil.rmtree(dest)
#     if return_file:
#         return os.abspath(min_e_file), energy
#     else:
#         return new_mol, energy