# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import traceback
import json
import subprocess
import copy
import shutil
import os # this is jsut vor OMP
import sys
sys.path.append('/home/julius/soft/GB-GA/')

from catalyst.utils import sdf2mol, draw3d, mol_from_xyz, vis_trajectory, Timer, hartree2kcalmol, Individual
from catalyst.make_structures import ConstrainedEmbedMultipleConfs, connect_cat_2d
from catalyst.xtb_utils import xtb_optimize, write_xtb_input_files 


prereactant_dummy = sdf2mol('/home/julius/soft/GB-GA/catalyst/structures/prereactant_dummy.sdf')
frag_energies = np.sum([-19.734652802142, -32.543971411432])

# %%
def prereactant_scoring(individual, args_list):
    n_confs, randomseed, timing_logger, warning_logger, directory, cpus_per_molecule = args_list
    warning = None
    try:
        energy = prereactant(cat=individual.rdkit_mol, gen_num=individual.idx[0], ind_num=individual.idx[1], n_confs=n_confs, randomseed=randomseed, numThreads=cpus_per_molecule, timing_logger=timing_logger, warning_logger=warning_logger, directory=directory) 
    except Exception as e:
        if warning_logger:
            warning_logger.warning(f'{individual.smiles}: {traceback.print_exc()}')
        else:
            print(f'{individual.smiles}: {traceback.print_exc()}')
        energy = None
        warning = str(e)
    return energy, warning

def prereactant(cat, ind_num, gen_num, n_confs=5, randomseed=101, numThreads=1, timing_logger=None, warning_logger=None, directory='.'):
    if timing_logger:
        t1 = Timer(logger=None)
        t1.start()
    ind_dir = f'G{gen_num:02d}_I{ind_num:02d}'
    cat_charge = Chem.GetFormalCharge(cat)
    prereactant2ds = connect_cat_2d(prereactant_dummy, cat)
    prereactant3d_energies = []
    prereactant3d_files = []
    for i, prereactant2d in enumerate(prereactant2ds): # for each constitutional isomer
        force_constant = 1000
        prereactant3d = ConstrainedEmbedMultipleConfs(mol=prereactant2d, core=prereactant_dummy, numConfs=int(n_confs), randomseed=int(randomseed), numThreads=int(numThreads), force_constant=int(force_constant))
        # xTB optimite TS
        prereactant3d_file, prereactant3d_energy = xtb_optimize(prereactant3d, method='gfnff', name=os.path.join(ind_dir, f'const_iso{i:03d}'), charge=cat_charge, constrains=None, scratchdir=directory, remove_tmp=False, return_file=True, numThreads=numThreads)
        # here should be a num frag check
        prereactant3d_energies.append(prereactant3d_energy)
        prereactant3d_files.append(prereactant3d_file)
    prereactant_energy = min(prereactant3d_energies)
    min_e_index = prereactant3d_energies.index(prereactant_energy)
    prereactant_file = prereactant3d_files[min_e_index] # lowest energy TS constitutional isomer
    # make gfn2 opt
    gfn2_opt_dir = os.path.join(os.path.dirname(os.path.dirname(prereactant_file)), 'gfn2_opt_prereactant')
    os.mkdir(gfn2_opt_dir)
    minE_pre_regioisomer_file = os.path.join(gfn2_opt_dir, 'minE_prereactant_isomer.xyz')
    shutil.move(prereactant_file, minE_pre_regioisomer_file)
    minE_pre_regioisomer_opt, gfn2_pre_energy = xtb_optimize(minE_pre_regioisomer_file, method='gfn2', constrains=None, name=None, charge=cat_charge, scratchdir=directory, remove_tmp=False, return_file=True, numThreads=numThreads)
    
    # splitting and calcualating cat for min E constitutional conformer
    cat_path = os.path.join(directory, os.path.join(os.path.dirname(os.path.dirname(minE_pre_regioisomer_opt)), 'catalyst'))
    os.mkdir(cat_path)
    cat_file = isolate_cat_from_xyz(minE_pre_regioisomer_opt, os.path.join(cat_path, 'cat.xyz'))
    cat_opt_file, cat_energy = xtb_optimize(cat_file, name=None, method='gfn2', charge=cat_charge, scratchdir=directory, remove_tmp=False,return_file=True, numThreads=numThreads)
    if timing_logger:
        elapsed_time = t1.stop()
        timing_logger.info(f'{Chem.MolToSmiles(cat)} : {elapsed_time:0.4f} seconds')
    return hartree2kcalmol(gfn2_pre_energy-cat_energy-frag_energies)


def isolate_cat_from_xyz(xyz_file, cat_xyz, num_atoms_scarfold=28):
    common_path = os.path.dirname(xyz_file)
    with open(xyz_file, 'r') as _file:
        with open(os.path.join(common_path, cat_xyz), 'w') as cat_file:
            for i, line in enumerate(_file):
                if i == 0:
                    old_num_atoms = int(line)
                    cat_file.write(f'{int(old_num_atoms-num_atoms_scarfold)}\n    catalyst\n')
                if i < num_atoms_scarfold+2:
                    pass
                else:
                    cat_file.write(f'{line}')
    return os.path.abspath(cat_xyz)

