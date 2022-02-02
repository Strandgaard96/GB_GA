# %%
from rdkit import Chem
from rdkit.Chem import AllChem

import sys

sys.path.append("/home/julius/soft/GB-GA/")
from catalyst.utils import sdf2mol, draw3d
from catalyst.gaussian_utils import out2mol


# %%    MAKEING THE TS DUMMY
ts = out2mol("/home/julius/thesis/sims/correlation/gfn2/mol002/gaussianTS/tsguess.out")
draw3d(ts)

# %%
quart_amine = Chem.MolFromSmarts("[#7X4;H0;D4;+1]")
quart_amine_id = ts.GetSubstructMatch(quart_amine)[0]
dummy = Chem.MolFromSmiles("*")
cat = Chem.MolFromSmarts(
    "[#7]12-[#6](-[#6](-[#7](-[#6](-[#6]-1(-[H])-[H])(-[H])-[H])-[#6](-[#6]-2(-[H])-[H])(-[H])-[H])(-[H])-[H])(-[H])-[H]"
)
# %%
ts_dummy = AllChem.ReplaceCore(ts, cat)

#%%
draw3d(ts_dummy)
writer = Chem.SDWriter("/home/julius/soft/GB-GA/catalyst/structures/ts_dummy.sdf")
writer.write(ts_dummy)

# %% Make the Reactant Dummy
from catalyst.make_structures import connect_cat_2d, ConstrainedEmbedMultipleConfs

reactant2 = sdf2mol("/home/julius/soft/GB-GA/catalyst/structures/reactant_dummy.sdf")
reactant = Chem.GetMolFrags(reactant2, asMols=True)[0]

emol = Chem.rdchem.EditableMol(reactant)
emol.RemoveAtom(28)
final_reactant = emol.GetMol()
final_reactant.GetAtomWithIdx(27).SetFormalCharge(-1)
Chem.SanitizeMol(final_reactant)
# %%
reactant2d = connect_cat_2d(final_reactant, Chem.MolFromSmiles("C1CN2CCN1CC2"))[0]
reactant3d = ConstrainedEmbedMultipleConfs(reactant2d, final_reactant, numConfs=5)
# %%
from catalyst.xtb_utils import xtb_optimize

reactant3d_opt, energy = xtb_optimize(
    reactant3d,
    charge=Chem.GetFormalCharge(reactant3d),
    scratchdir=".",
    name="prereactant",
    remove_tmp=False,
)
# %%
quart_amine = Chem.MolFromSmarts("[#7X4;H0;D4;+1]")
quart_amine_id = reactant3d_opt.GetSubstructMatch(quart_amine)[0]
dummy = Chem.MolFromSmiles("*")
cat = Chem.MolFromSmarts(
    "[#7]12-[#6](-[#6](-[#7](-[#6](-[#6]-1(-[H])-[H])(-[H])-[H])-[#6](-[#6]-2(-[H])-[H])(-[H])-[H])(-[H])-[H])(-[H])-[H]"
)
# %%
prereactant_dummy = AllChem.ReplaceCore(reactant3d_opt, cat)
draw3d(prereactant_dummy)
# %%
writer = Chem.SDWriter(
    "/home/julius/soft/GB-GA/catalyst/structures/prereactant_dummy.sdf"
)
writer.write(prereactant_dummy)
# %%
