import os
import numpy as np
import subprocess

from rdkit import Chem
from rdkit.Chem import GetPeriodicTable

import sys
sys.path.append('/home/julius/soft/GB-GA/')
from catalyst.utils import mol_from_xyz

def write_gaussian_input_file(fragment, fragment_name, command='opt freq b3lyp/6-31+g(d,p) scrf=(smd,solvent=methanol) empiricaldispersion=gd3', mem=4, cpus=4, constr=None):
    number_of_atoms = fragment.GetNumAtoms()
    charge = Chem.GetFormalCharge(fragment)
    multiplicity = 1
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()] 
    file_name = fragment_name +".com"
    chk_file = fragment_name + '.chk'
    conformers = fragment.GetConformers()
    if not conformers:
        raise Exception('Mol is not embedded.')
    if not conformers[0].Is3D:
        raise Exception('Mol is not 3D.')
    constr_array = np.zeros(fragment.GetNumAtoms(),dtype=int)
    if constr:
        constr_atoms = read_inp(inp_file=constr)
        for atom in constr_atoms:
            constr_array[atom] = -1
    for i,conf in enumerate(fragment.GetConformers()):
        with open(file_name, "w") as file:
            file.write(f'%mem={mem}GB\n%nprocshared={cpus}\n%chk={chk_file}\n# {command}\n\n{fragment_name}\n\n{charge} {multiplicity}\n')
            for atom,symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                binary_constr = constr_array[atom]
                line = " ".join((' '+symbol, str(binary_constr), str(p.x),str(p.y),str(p.z),"\n"))
                file.write(line)
            file.write(f'\n')
    return file_name

def compute_dft(mol, name=None, directory='.', command='opt freq b3lyp/6-31+g(d,p) scrf=(smd,solvent=methanol) empiricaldispersion=gd3', constr=None, cpus=4):
    orgdir= os.getcwd()
    if not name:
        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    dir = os.path.join(directory, name)
    os.mkdir(dir)
    os.chdir(dir)
    file_name = write_gaussian_input_file(mol, 'tsguess', constr=constr, command=command, cpus=cpus)
    # out = shell(f'/opt/gaussian/g16/legacy/g16/g16 < {file_name}.com > dft.out', shell=False)
    os.system(f'submit_gaussian_legacy {file_name}')
    os.chdir(orgdir)

def make_gaussian_constrained_opt(xyz_file, bond_pairs_changed, charge):
    """
    This function prepares a Gaussian input file for a constrained
    optimization freezing the bonds broken or created during the reaction.
    The .com file is submitted to slurm using 'submit_gaus16' which returns a
    jobid used to monitor the job
    """
    n_frozen_bonds = len(bond_pairs_changed)
    com_file = xyz_file[:-4]+'_opt.com'
    with open(com_file, 'w') as _file:
        _file.write('%mem=1GB'+'\n')
        _file.write('%nprocshared=1'+'\n')
        _file.write("#opt external='~/soft/gau_xtb/xtb.sh' int=ultrafine ")# geom(modredundant, gic) ")
        _file.write('scf=(maxcycles=1024)'+'\n\n')
        _file.write(f'constroptxtb'+'\n\n')
        _file.write(f'{charge} 1'+'\n') #hardcoded to neutral singlet, change if needed
        with open(xyz_file, 'r') as file_in:
            lines = file_in.readlines()
            lines = lines[2:]
            _file.writelines(lines)
        _file.write('\n')
        # for i, bond_pair in zip(range(n_frozen_bonds), bond_pairs_changed):
        #     _file.write('bond'+str(i)+'(freeze)=B('+str(bond_pair[0]+1)+',')
        #     _file.write(str(bond_pair[1]+1)+')'+'\n')
        # _file.write('\n')

        p = subprocess.Popen(f'submit_gaussian_legacy {com_file}', cwd=os.path.dirname(com_file), shell=True)
        out_file = com_file[:-4]+".out"
    return out_file

def gauxtb(com_file):
    orgdir= os.getcwd()
    dir = os.path.dirname(com_file)
    os.chdir(dir)
    # out = shell(f'/opt/gaussian/g16/legacy/g16/g16 < {file_name}.com > dft.out', shell=False)
    # os.system(f'submit_gaussian_legacy {com_file}')
    p = subprocess.Popen(f'submit_gaussian_legacy {com_file}', shell=True)
    _, _ = p.communicate()
    os.chdir(orgdir)

def subsequent_dft(oldchk, command='opt=(TS,noeigentest,calcfc) b3lyp/6-31+g(d,p) scrf=(smd,solvent=methanol) empiricaldispersion=gd3 Geom=AllCheckpoint', mem=2, cpus=4):
    name = os.path.basename(oldchk).split('.')[0]
    file_name = os.path.join(os.path.dirname(oldchk), f'{name}631.com')
    with open(file_name, "w") as file:
        file.write(f'%mem={mem}GB\n%nprocshared={cpus}\n%oldchk={oldchk}\n%chk={name}631.chk\n# {command}\n\n\n')
    # out = shell(f'/opt/gaussian/g16/legacy/g16/g16 < {file_name}.com > dft.out', shell=False)
    orgdir= os.getcwd()
    os.chdir(os.path.dirname(file_name))
    p = subprocess.Popen(f'submit_gaussian_legacy_42 {name}631.com', shell=True)
    _, _ = p.communicate()
    os.chdir(orgdir)

def compute_freq(oldchk, directory='.', command="freq external='~/soft/gau_xtb/xtb.sh' Geom=AllCheckpoint", mem=2, cpus=4):
    file_name = os.path.join(directory, 'tsfreq.com')
    if os.path.exists(file_name):
            os.remove(file_name)
    with open(file_name, "w") as file:
        file.write(f'%mem={mem}GB\n%nprocshared={cpus}\n%oldchk={oldchk}\n%chk=tsfreq.chk\n# {command}\n\n\n')
    # out = shell(f'/opt/gaussian/g16/legacy/g16/g16 < {file_name}.com > dft.out', shell=False)
    orgdir= os.getcwd()
    os.chdir(os.path.dirname(file_name))
    print(os.getcwd())
    p = subprocess.Popen(f'submit_gaussian_legacy42 tsfreq.com', shell=True)
    _, _ = p.communicate()
    os.chdir(orgdir)

def get_energy(out_file):
    with open(out_file, 'r') as ofile:
        line = ofile.readline()
        while line:
            if 'Recovered energy=' in line: # this is for externally calcualted energy
                energy = float(line.split('=')[1].split(' ')[1])
            if 'SCF Done:' in line:# this is for internally calculated energy
                energy = float(line. split(':')[1]. split('=')[1].split('A.U.')[0])
            line = ofile.readline()
    return energy

def extract_optimized_structure(out_file, return_mol=True, returnE=False):
    """
    After waiting for the constrained optimization to finish, the
    resulting structure from the constrained optimization is
    extracted and saved as .xyz file ready for TS optimization.
    """
    optimized_xyz_file = out_file[:-4]+".xyz"
    optimization_complete = False
    atom_labels = []
    pse = GetPeriodicTable()
    with open(out_file, 'r') as ofile:
        line = ofile.readline()
        while line:
            if 'NAtoms=' in line:
                n_atoms = int(line.split('=')[1].split(' ')[-2])
            if 'Recovered energy=' in line: # this is for externally calcualted energy
                energy = float(line.split('=')[1].split(' ')[1])
            if 'SCF Done:' in line:# this is for internally calculated energy
                energy = float(line. split(':')[1]. split('=')[1].split('A.U.')[0])
            if 'Stationary point found' in line:
                optimization_complete = True
            if optimization_complete and 'Standard orientation' in line:
                coordinates = np.zeros((n_atoms, 3))
                for i in range(5):
                    line = ofile.readline()
                for i in range(n_atoms):
                    atom_labels.append(pse.GetElementSymbol(int(line.split()[1])))
                    coordinates[i, :] = np.array(line.split()[-3:])
                    line = ofile.readline()
            line = ofile.readline()
    with open(optimized_xyz_file, 'w') as _file:
        _file.write(str(n_atoms)+'\n\n')
        for i in range(n_atoms):
            _file.write(atom_labels[i])
            for j in range(3):
                _file.write(' '+"{:.5f}".format(coordinates[i, j]))
            _file.write('\n')

    if return_mol:
        if returnE:
            return mol_from_xyz(optimized_xyz_file), energy
        else:
            return mol_from_xyz(optimized_xyz_file)
    else:
        if returnE:
            return optimized_xyz_file, energy
        else:
            return optimized_xyz_file

def write_irc_input_file(old_chk_file, name, direction, mem=4, cpus=4, constr=None):
    command = f"ircmax=({direction},calcfc,maxpoints=300,recalc=3) external='~/soft/gau_xtb/xtb.sh' int=ultrafine Geom=Checkpoint"
    file_name = name +".com"
    new_chk_file = name + '.chk'
    with open(file_name, "w") as file:
        file.write(f'%mem={mem}GB\n%nprocshared={cpus}\n%oldchk={old_chk_file}\n%chk={new_chk_file}\n# {command}\n\n{os.path.basename(name)}\n\n0 1\n')
        file.write(f'\n')
    return file_name

def make_irc(oldchk, moldir):
    ts_chk = os.path.join(moldir, 'gaussianTS/tsguess.chk')
    irc_dir = os.path.join(moldir, 'irc')
    os.mkdir(irc_dir)

    reverse = os.path.join(irc_dir, 'reverse')
    os.mkdir(reverse)
    reverse_com = write_irc_input_file(ts_chk, os.path.join(reverse,'revIRC'), 'reverse')
    orgpwd = os.getcwd()
    os.chdir(reverse)
    com_file = os.path.basename(reverse_com)
    os.system(f'submit_gaussian_legacy {com_file}')
    os.chdir(orgpwd)

    forward = os.path.join(irc_dir, 'forward')
    os.mkdir(forward)
    forward_com = write_irc_input_file(ts_chk, os.path.join(forward,'forIRC'), 'forward')
    orgpwd = os.getcwd()
    os.chdir(forward)
    com_file = os.path.basename(forward_com)
    os.system(f'submit_gaussian_legacy {com_file}')
    os.chdir(orgpwd)

