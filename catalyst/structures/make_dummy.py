# %%
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
sys.path.append('/home/julius/soft/GB-GA/')
from catalyst.utils import sdf2mol, out2mol, draw3d


# %%    MAKEING THE TS DUMMY
ts = out2mol('/home/julius/thesis/sims/correlation/gfn2/mol002/gaussianTS/tsguess.out')
draw3d(ts)

# %%
quart_amine = Chem.MolFromSmarts('[#7X4;H0;D4;+1]')
quart_amine_id = ts.GetSubstructMatch(quart_amine)[0]
dummy = Chem.MolFromSmiles('*')
cat = Chem.MolFromSmarts('[#7]12-[#6](-[#6](-[#7](-[#6](-[#6]-1(-[H])-[H])(-[H])-[H])-[#6](-[#6]-2(-[H])-[H])(-[H])-[H])(-[H])-[H])(-[H])-[H]')
# %%
ts_dummy = AllChem.ReplaceCore(ts, cat)

#%%
draw3d(ts_dummy)
writer = Chem.SDWriter('/home/julius/soft/GB-GA/catalyst/structures/ts_dummy.sdf')
writer.write(ts_dummy)