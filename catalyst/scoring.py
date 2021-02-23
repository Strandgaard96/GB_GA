# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import copy
import shutil
import os # this is jsut vor OMP
import sys
sys.path.append('/home/julius/soft/GB-GA/')
from catalyst.utils import sdf2mol
from catalyst.make_structures import ConstrainedEmbedMultipleConfs, connect_cat_2d, check_num_frags
from catalyst.xtb_utils import xtb_optimize

ts_dummy = sdf2mol('/home/julius/soft/GB-GA/catalyst/structures/ts_dummy.sdf')
frag_energies = np.sum([-8.232710038092, -19.734652802142, -32.543971411432]) # 34 atoms

numThreads = int(sys.argv[-2]) # this is the total im using, n_cpuy is threads per molecule
directory = sys.argv[-1]

# %%
def final_scoring(mol, args_list): # to be used in calculate_scores_parallel(population,final_scoring,[gen_num, n_confs, randomseed],n_cpus)
    ind_num, gen_num, n_confs, randomseed, n_cpus = args_list
    energy = activation_barrier(cat=mol, gen_num=gen_num, ind_num=ind_num, n_confs=n_confs, randomseed=randomseed, numThreads=n_cpus) 
    return energy

def activation_barrier(cat, ind_num, gen_num, n_confs=5, randomseed=101, numThreads=1):
    ts2ds = connect_cat_2d(ts_dummy, cat)
    ts3d_energies = []
    ts3d_structures = []
    for i, ts2d in enumerate(ts2ds): # for each constitutional isomer
        print(f'Doing Conformer {i}')
        # Embedding while trying several different force constants
        angles_ok = False
        bonds_ok = False
        force_constant = 100000
        max_tries = 4
        tries = 0
        while not angles_ok or not bonds_ok and tries < max_tries:
            ts2d_copy = copy.deepcopy(ts2d)
            ts3d = ConstrainedEmbedMultipleConfs(mol=ts2d_copy, core=ts_dummy, numConfs=n_confs, randomseed=randomseed, numThreads=numThreads, force_constant=force_constant)
            if compare_angles(ts3d, ts_dummy, threshold=2):
                angles_ok = True
            else:
                force_constant = force_constant*10
            if bonds_OK(ts3d, threshold=2):
                bonds_ok = True
            else:
                force_constant = force_constant/10
            tries += 1
            if tries == max_tries:
                raise Exception(f'Embedding of {Chem.MolToSmiles(ts2d)} was not successful in {max_tries} tries.')
        # xTB optimite TS
        ts3d_opt, ts3d_energy = xtb_optimize(ts3d, name=f'G{gen_num:02d}_I{ind_num:02d}/const_iso{i:03d}', constrains='/home/julius/soft/GB-GA/catalyst/constr_opt.inp', scratchdir=directory, remove_tmp=False, numThreads=numThreads)
        if check_num_frags(ts3d_opt, 3): # if molecue has 3 fragments after optimization 
            ts3d_energies.append(ts3d_energy)
            ts3d_structures.append(ts3d_opt)
        
    # make_tarfile('log.tar', directory)
    # exe_dir = os.getcwd()
    # print(exe_dir)
    # shutil.move(os.path.join(directory, 'log.tar'), os.path.join(exe_dir, 'log.tar'))
    try:
        min_e_index = ts3d_energies.index(min(ts3d_energies))
        ts = ts3d_structures[min_e_index]
    except:
        print(f'xTB optimization led to missmatch in number of fragments for {Chem.MolToSmiles(ts2d)}')
        return -100000
    cat3d = isolate_cat(ts, ts_dummy)
    if cat3d is not None:
        cat_opt, cat_energy = xtb_optimize(cat3d, name=f'G{gen_num:02d}_I{ind_num:02d}/catalyst', scratchdir=directory, remove_tmp=False, numThreads=numThreads)
    else:
        print(f'Was not able to isolate cat from {Chem.MolToSmiles(ts)}, it should be {Chem.MolToSmiles(cat)}')
        cat_energy = -1000
    return min(ts3d_energies)-cat_energy-frag_energies

def isolate_cat(mol, scarfold):
    if not mol.HasSubstructMatch(scarfold):
        print(f'Optimization of {Chem.MolToSmiles(mol)} did not go right')
        return None
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
    directory = '.'
    numThreads = 1
    n_confs = 1
    # randomseed = 123 # das hat nen charegd cat
    # randomseed = 101 # das hat nen fluo und das gibt probleme bei xyzmol und xyt opt
    randomseed = 321
    generation = 0

    random.seed(randomseed)
    np.random.seed(randomseed)

    population = make_initial_population(25, '/home/julius/soft/GB-GA/ZINC_1000_amines.smi')
    # scoring_args = [generation, n_confs, randsomseed]
    # output = calculate_scores_parallel(population, final_scoring, scoring_args, numThreads)
    # t.stop()
    # print(output)

    
    for i, mol in enumerate(population):
        with open('./log.log', 'w') as _file:
            _file.write(f'Processing {Chem.MolToSmiles(mol)}\n')
        print(Chem.MolToSmiles(mol))
        energy = activation_barrier(cat=mol, gen_num=0, ind_num=i, n_confs=n_confs, randomseed=randomseed, numThreads=numThreads)
        print(energy)
        break



# %%

