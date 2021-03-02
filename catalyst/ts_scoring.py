# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import logging
import copy
import shutil
import os # this is jsut vor OMP
import sys
sys.path.append('/home/julius/soft/GB-GA/')
from catalyst.utils import sdf2mol, draw3d, mol_from_xyz, vis_trajectory, Timer
from catalyst.make_structures import ConstrainedEmbedMultipleConfsMultipleFrags, connect_cat_2d
from catalyst.xtb_utils import xtb_optimize, write_xtb_input_files 
ts_dummy = sdf2mol('/home/julius/soft/GB-GA/catalyst/structures/ts_dummy.sdf')
frag_energies = np.sum([-8.232710038092, -19.734652802142, -32.543971411432]) # 34 atoms

# numThreads = int(sys.argv[-2]) # this is the total im using, n_cpuy is threads per molecule
# directory = sys.argv[-1]
# directory = '/home/julius/thesis/sims/ts_embed_scoring'

# %%
def ts_scoring(mol, args_list): # to be used in calculate_scores_parallel(population,final_scoring,[gen_num, n_confs, randomseed],n_cpus)
    ind_num, gen_num, n_confs, randomseed, timing_logger, warning_logger, directory, n_cpus = args_list
    energy = activation_barrier(cat=mol, gen_num=gen_num, ind_num=ind_num, n_confs=n_confs, randomseed=randomseed, numThreads=n_cpus, timing_logger=timing_logger, warning_logger=warning_logger, directory=directory) 
    return energy

def activation_barrier(cat, ind_num, gen_num, n_confs=5, randomseed=101, numThreads=1, timing_logger=None, warning_logger=None, directory='.'):
    if timing_logger:
        t1 = Timer(logger=None)
        t1.start()
    ind_dir = f'G{gen_num:02d}_I{ind_num:02d}'
    cat_charge = Chem.GetFormalCharge(cat)
    ts2ds = connect_cat_2d(ts_dummy, cat)
    ts3d_energies = []
    ts3d_files = []
    for i, ts2d in enumerate(ts2ds): # for each constitutional isomer
        angles_ok = False
        bonds_ok = False
        force_constant = 5000
        max_tries = 8
        tries = 0
        while not angles_ok or not bonds_ok and tries < max_tries:
            ts2d_copy = copy.deepcopy(ts2d)
            ts3d = ConstrainedEmbedMultipleConfsMultipleFrags(mol=ts2d_copy, core=ts_dummy, numConfs=n_confs, randomseed=randomseed, numThreads=numThreads, force_constant=force_constant)
            # ts3d = ConstrainedEmbedMultipleConfs(mol=ts2d_copy, core=ts_dummy, numConfs=n_confs, randomseed=randomseed, numThreads=numThreads, force_constant=force_constant)
            if compare_angles(ts3d, ts_dummy, threshold=5):
                angles_ok = True
            else:
                force_constant = force_constant*2
            if bonds_OK(ts3d, threshold=2):
                bonds_ok = True
            else:
                force_constant = force_constant/2
            tries += 1
            if tries == max_tries:
                if warning_logger:
                    warning_logger.warning(f'Embedding of {Chem.MolToSmiles(ts2d)} was not successful in {max_tries} tries with final forceconstant={force_constant}')
                else:
                    print(f'Embedding of {Chem.MolToSmiles(ts2d)} was not successful in {max_tries} tries with final forceconstant={force_constant}')
                break
                # raise Exception(f'Embedding of {Chem.MolToSmiles(ts2d)} was not successful in {max_tries} tries with final forceconstant={force_constant}.')
        # xTB optimite TS
        ts3d_file, ts3d_energy = xtb_optimize(ts3d, method='gfnff', name=os.path.join(ind_dir, f'const_iso{i:03d}'), charge=cat_charge, constrains='/home/julius/thesis/data/constr.inp', scratchdir=directory, remove_tmp=False, return_file=True, numThreads=numThreads, warning_logger=warning_logger)
        # here should be a num frag check
        ts3d_energies.append(ts3d_energy)
        ts3d_files.append(ts3d_file)
    # try:
    ts_energy = min(ts3d_energies)
    min_e_index = ts3d_energies.index(ts_energy)
    ts_file = ts3d_files[min_e_index] # lowest energy TS constitutional isomer
    # make gfn2 opt
    gfn2_opt_dir = os.path.join(os.path.abspath(ind_dir), 'gfn2_opt_TS')
    os.mkdir(gfn2_opt_dir)
    minE_TS_regioisomer_file = os.path.join(gfn2_opt_dir, 'minE_TS_isomer.xyz')
    shutil.move(ts_file, minE_TS_regioisomer_file)
    print(minE_TS_regioisomer_file, gfn2_opt_dir, ts_file)
    minE_TS_regioisomer_opt, gfn2_ts_energy = xtb_optimize(minE_TS_regioisomer_file, method='gfn2', constrains='/home/julius/thesis/data/constr.inp', name=None, charge=cat_charge, scratchdir=directory, remove_tmp=False,return_file=True, numThreads=numThreads, warning_logger=warning_logger)
    # except:
    #     if warning_logger:
    #         warning_logger.warning(f'xTB optimization led to missmatch in number of fragments for {Chem.MolToSmiles(ts2d)}')
    #     else:
    #         print(f'xTB optimization led to missmatch in number of fragments for {Chem.MolToSmiles(ts2d)}')
    #     return(-100000)
    
    # splitting and calcualating cat for min E constitutional conformer
    cat_path = os.path.join(directory, os.path.join(os.path.dirname(minE_TS_regioisomer_opt), 'catalyst'))
    os.mkdir(cat_path)
    cat_file = isolate_cat_from_xyz(minE_TS_regioisomer_opt, os.path.join(cat_path, 'cat.xyz'))
    cat_opt_file, cat_energy = xtb_optimize(cat_file, name=None, method='gfn2', charge=cat_charge, scratchdir=directory, remove_tmp=False,return_file=True, numThreads=numThreads, warning_logger=warning_logger)
    if timing_logger:
        elapsed_time = t1.stop()
        timing_logger.info(f'{Chem.MolToSmiles(cat)} : {elapsed_time:0.4f} seconds')
    return gfn2_ts_energy-cat_energy-frag_energies

def isolate_cat(mol, scarfold):
    cat = AllChem.ReplaceCore(mol, scarfold, replaceDummies=False)
    for atom in cat.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_atom = atom
            break
    for atom in dummy_atom.GetNeighbors():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
            quart_amine = atom
            break
    quart_amine.SetFormalCharge(0)
    edmol = Chem.EditableMol(cat)
    edmol.RemoveAtom(dummy_atom.GetIdx())
    cat = edmol.GetMol()
    Chem.SanitizeMol(cat)
    return cat

def isolate_cat_from_xyz(xyz_file, cat_xyz, num_atoms_scarfold=34):
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


# def positions_OK(mol):


def bonds_OK(mol, threshold):
    mol_confs = mol.GetConformers()
    for conf in mol_confs:
        for bond in mol.GetBonds():
            length = AllChem.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if length > threshold:
                return False
    return True

def compare_angles(mol, ref, ids=(1,11,29), threshold=2):
    ref_conf = ref.GetConformer()
    ref_angle = AllChem.GetAngleDeg(ref_conf, ids[0], ids[1], ids[2])
    mol_confs = mol.GetConformers()
    mol_angles = []
    for mol_conf in mol_confs:
        mol_angle = AllChem.GetAngleDeg(mol_conf, ids[0], ids[1], ids[2])
        mol_angles.append(mol_angle)
    for angle in mol_angles:
        if abs(ref_angle - angle) < threshold:
            return True
        else:
            return False

import tarfile
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

# %%
if __name__ == '__main__':
    import random
    from catalyst.utils import mols_from_smi_file, Timer
    from scoring_functions import calculate_scores_parallel
    # population = mols_from_smi_file('/home/julius/thesis/data/QD_cats.smi')
    from GB_GA import make_initial_population

    t = Timer()
    t.start()
    directory = '/home/julius/thesis/sims/ts_embed_scoring'
    numThreads = 4
    n_confs = 3
    # randomseed = 123 # das hat nen charegd cat
    randomseed = 101 # das hat nen fluo und das gibt probleme bei xyzmol und xyt opt
    # randomseed = 321
    generation = 0

    random.seed(randomseed)
    np.random.seed(randomseed)

    population = make_initial_population(25, '/home/julius/soft/GB-GA/ZINC_1000_amines.smi')
    # scoring_args = [generation, n_confs, randsomseed]
    # output = calculate_scores_parallel(population, final_scoring, scoring_args, numThreads)
    # t.stop()
    # print(output)

    import logging

    formatter = logging.Formatter('%(message)s')
    # Timing Logger
    timing_logger = logging.getLogger(__name__)
    timing_logger.setLevel(logging.INFO)    

    timing_file_handler = logging.FileHandler('scoring_timings.log')
    timing_file_handler.setFormatter(formatter)

    timing_logger.addHandler(timing_file_handler)

    timing_logger.info(f'# Running on {numThreads} cores')

    for i, mol in enumerate(population):
        activation_barrier(cat=mol, ind_num=i, gen_num=0, n_confs=2, randomseed=101, numThreads=4, logger=timing_logger)

# %%
analyse = False
if analyse:
    mols = []
    cat_energies = []
    ts_energies = []
    with open('/home/julius/soft/GB-GA/catalyst/log.log', 'r') as log:
        for line in log:
            mol, cat_energy, ts_energy = line.split(',')
            mols.append(mol)
            cat_energies.append(float(cat_energy))
            ts_energies.append(float(ts_energy))
    cat_energies = np.array(cat_energies)
    ts_energies = np.array(ts_energies)

