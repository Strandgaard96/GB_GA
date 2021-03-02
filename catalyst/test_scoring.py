# %%
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem

# import copy
# import shutil
# import os # this is jsut vor OMP
import random
import numpy as np
import sys
sys.path.append('/home/julius/soft/GB-GA/')
# from catalyst.utils import sdf2mol, draw3d, mol_from_xyz, vis_trajectory
# from catalyst.make_structures import ConstrainedEmbedMultipleConfsMultipleFrags, connect_cat_2d, check_num_frags
# from catalyst.xtb_utils import xtb_optimize, write_xtb_input_files 
from catalyst.ts_scoring import ts_scoring
from scoring_functions import calculate_scores_parallel
from GB_GA import make_initial_population

# %%

numThreads = int(sys.argv[-2]) # this is the total im using
directory = sys.argv[-1]

n_confs = 3
generation = 0
randomseed = 101

random.seed(randomseed)
np.random.seed(randomseed)

population = make_initial_population(4, '/home/julius/soft/GB-GA/ZINC_1000_amines.smi')

import logging
formatter = logging.Formatter('%(message)s')
timing_logger = logging.getLogger(__name__)
timing_logger.setLevel(logging.INFO)
timing_file_handler = logging.FileHandler('scoring_timings.log')
timing_file_handler.setFormatter(formatter)
timing_logger.addHandler(timing_file_handler)
timing_logger.info(f'# Running on {numThreads} cores')

scoring_args = [generation, n_confs, randomseed, timing_logger, directory]
scores = calculate_scores_parallel(population=population, function=ts_scoring, scoring_args=scoring_args, n_cpus=numThreads)
print(scores)
# %%
if False:
    from rdkit import Chem
    from catalyst.utils import draw3d
    from catalyst.xtb_utils import get_energy_from_xtb_sp
    randomseed = 101

    random.seed(randomseed)
    np.random.seed(randomseed)
    population = make_initial_population(4, '/home/julius/soft/GB-GA/ZINC_1000_amines.smi')
    for i, mol in enumerate(population):
        ts_file = f'/home/julius/thesis/sims/ts_embed_scoring/G00_I0{i}/const_iso000/min_e_conf.xyz'
        print(Chem.MolToSmiles(mol), get_energy_from_xtb_sp(ts_file))
        draw3d(ts_file)
# %%
