from rdkit import Chem
from rdkit.Chem import AllChem
from xyz2mol.xyz2mol import read_xyz_file, xyz2mol

import os
import random
import shutil
import string
import subprocess


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

def xtb_optimize(mol, name=None, constrains=None, method='gfn2', solvent='alpb methanol', opt_level='tight', scratchdir='/home/julius/thesis/sims/scratch', remove_tmp=True, return_file=False, numThreads=1):
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
    os.makedirs(dest)
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
        os.environ['OMP_NUM_THREADS'] = f'{numThreads},1'
        os.environ['OMP_STACKSIZE'] = '4G'
        print(f"right before calling subprosess {os.environ['OMP_NUM_THREADS']}")
        p = subprocess.Popen(
            f'/home/julius/soft/xtb-6.3.3/bin/xtb --{method} {xyz_file} --opt {opt_level} {constrains_input} {solvent_input} --json > out.out', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        out_file = 'xtbopt.xyz'
        try:
            energy = get_energy_from_xtb_sp(out_file)
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
    if return_file:
        return os.abspath(min_e_file), energy
    else:
        return new_mol, energy