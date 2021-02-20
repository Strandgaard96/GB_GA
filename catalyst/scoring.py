# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import os # this is jsut vor OMP
import sys
sys.path.append('/home/julius/soft/GB-GA/')
from catalyst.utils import sdf2mol
from catalyst.make_structures import ConstrainedEmbedMultipleConfs, connect_cat_2d
from catalyst.xtb_utils import xtb_optimize

ts_dummy = sdf2mol('/home/julius/soft/GB-GA/catalyst/structures/ts_dummy.sdf')
frag_energies = np.sum([-8.232710038092, -19.734652802142, -32.543971411432]) # 34 atoms

numThreads = int(sys.argv[-2]) # this is the total im using, n_cpuy is threads per molecule
directory = sys.argv[-1]
# %%
def final_scoring(mol, args_list): # to be used in calculate_scores_parallel(population,final_scoring,[gen_num, n_confs, randomseed],n_cpus)
    os.environ['OMP_NUM_THREADS'] = f'{numThreads},1'
    population_size = 6
    ind_num, gen_num, n_confs, randomseed, n_cpus = args_list
    energy = activation_barrier(cat=mol, gen_num=gen_num, ind_num=ind_num, n_confs=n_confs, randomseed=randomseed, numThreads=1)
    return energy

def activation_barrier(cat, ind_num, gen_num, n_confs=5, randomseed=101, numThreads=1):
    ts2ds = connect_cat_2d(ts_dummy, cat)
    ts3d_energies = []
    ts3d_structures = []
    for i, ts2d in enumerate(ts2ds): # for each constitutional isomer
        ts3d = ConstrainedEmbedMultipleConfs(mol=ts2d, core=ts_dummy, numConfs=n_confs, randomseed=randomseed, numThreads=numThreads)
        ts3d_opt, ts3d_energy = xtb_optimize(ts3d, name=f'G{gen_num:02d}_I{ind_num:02d}/const_iso{i:03d}', constrains='/home/julius/soft/GB-GA/catalyst/constr_opt.inp', scratchdir=directory, remove_tmp=False, numThreads=numThreads)
        ts3d_energies.append(ts3d_energy)
        ts3d_structures.append(ts3d_opt)
    min_e_index = ts3d_energies.index(min(ts3d_energies))
    ts = ts3d_structures[min_e_index] # maybe write min constitutional isomer min conformer out in parentparent directory (GX_IY)
    cat3d = isolate_cat(ts, ts_dummy)
    if cat3d is not None:
        cat_opt, cat_energy = xtb_optimize(cat3d, name=f'G{gen_num:02d}_I{ind_num:02d}/catalyst', scratchdir=directory, remove_tmp=False, numThreads=numThreads)
    else:
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

# %%
if __name__ == '__main__':
    from catalyst.utils import mols_from_smi_file
    from scoring_functions import calculate_scores_parallel
    population = mols_from_smi_file('/home/julius/thesis/data/QD_cats.smi')
    # n_cpus = 4
    n_confs = 1
    randsomseed = 100
    generation = 0
    print(f"before: {os.environ['OMP_NUM_THREADS']}")
    scoring_args = [generation, n_confs, randsomseed]
    output = calculate_scores_parallel(population, final_scoring, scoring_args, numThreads)
    print(f"nach rechngun: {os.environ['OMP_NUM_THREADS']}")
    os.environ['OMP_NUM_THREADS'] = '1'
    print(f"After: {os.environ['OMP_NUM_THREADS']}")
    print(output)
    # for i, mol in enumerate(mols):
    #     energy = activation_barrier(cat=mol, gen_num=0, ind_num=i, n_confs=15, randomseed=101, numThreads=numThreads)
    #     print(energy)


# %%
