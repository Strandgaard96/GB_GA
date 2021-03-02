# %%
from rdkit import Chem
from rdkit.Chem import AllChem

# %% 
def connect_cat_2d(mol_with_dummy, cat):
    """Replaces Dummy Atom [*] in Mol with Cat via tertiary Amine, return list of all possible regioisomers"""
    dummy = Chem.MolFromSmiles('*')
    mols = []
    cat = Chem.AddHs(cat)
    tert_amines = cat.GetSubstructMatches(
        Chem.MolFromSmarts('[#7X3;H0;D3;!+1]'))
    for amine in tert_amines:
        mol = AllChem.ReplaceSubstructs(
            mol_with_dummy, dummy, cat, replacementConnectionPoint=amine[0])[0]
        quart_amine = mol.GetSubstructMatch(
            Chem.MolFromSmarts('[#7X4;H0;D4;!+1]'))[0]
        mol.GetAtomWithIdx(quart_amine).SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        mol.RemoveAllConformers()
        mols.append(mol)
    return mols

def frags2bonded(mol, atoms2join=((1,11), (11,29))):
    make_bonded = Chem.EditableMol(mol)
    for atoms in atoms2join:
        i, j = atoms
        make_bonded.AddBond(i,j)
    mol_bonded = make_bonded.GetMol()
    # Chem.SanitizeMol(mol_bonded)
    return mol_bonded

def bonded2frags(mol, atoms2frag=((1,11), (11,29))):
    make_frags = Chem.EditableMol(mol)
    for atoms in atoms2frag:
        i, j = atoms
        make_frags.RemoveBond(i,j)
    mol_frags = make_frags.GetMol()
    # Chem.SanitizeMol(mol_frags)
    return mol_frags

def ConstrainedEmbedMultipleConfs(mol, core, numConfs=10, useTethers=True, coreConfId=-1, randomseed=2342,
                     getForceField=AllChem.UFFGetMoleculeForceField, numThreads=1, force_constant=1e3):
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    if "." in Chem.MolToSmiles(mol):
        cids = AllChem.EmbedMultipleConfs(mol=mol, numConfs=numConfs, randomSeed=randomseed, numThreads=numThreads)  # jhj
    else:
        cids = AllChem.EmbedMultipleConfs(
            mol=mol, numConfs=numConfs, coordMap=coordMap, randomSeed=randomseed, numThreads=numThreads)

    cids = list(cids)
    if len(cids) == 0:
        raise ValueError('Could not embed molecule.')

    algMap = [(j, i) for i, j in enumerate(match)]
    
    if not useTethers:
        # clean up the conformation
        for cid in cids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid, ignoreInterfragInteractions=False)
            for i, idxI in enumerate(match):
                for j in range(i + 1, len(match)):
                    idxJ = match[j]
                    d = coordMap[idxI].Distance(coordMap[idxJ])
                    ff.AddDistanceConstraint(idxI, idxJ, d, d, force_constant)
            ff.Initialize()
            n = 4
            more = ff.Minimize()
            while more and n:
                more = ff.Minimize()
                n -= 1
            # rotate the embedded conformation onto the core:
            rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        for cid in cids:
            rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid, ignoreInterfragInteractions=False)
            conf = core.GetConformer()
            for i in range(core.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
                ff.AddDistanceConstraint(pIdx, match[i], 0, 0, force_constant)
            ff.Initialize()
            n = 4
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            while more and n:
                more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                n -= 1
            # realign
            rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    return mol

def ConstrainedEmbedMultipleConfsMultipleFrags(mol, core, numConfs=10, useTethers=True, coreConfId=-1, randomseed=2342,
                     getForceField=AllChem.UFFGetMoleculeForceField, numThreads=1, force_constant=1e3):
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    mol_bonded = frags2bonded(mol)
    cids = AllChem.EmbedMultipleConfs(
            mol=mol_bonded, numConfs=numConfs, coordMap=coordMap, randomSeed=randomseed, numThreads=numThreads)
    mol = bonded2frags(mol_bonded)
    Chem.SanitizeMol(mol)

    cids = list(cids)
    if len(cids) == 0:
        raise ValueError('Could not embed molecule.')

    algMap = [(j, i) for i, j in enumerate(match)]

    # rotate the embedded conformation onto the core:
    for cid in cids:
        rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid, ignoreInterfragInteractions=False)
        for i, _ in enumerate(match):
            ff.UFFAddPositionConstraint(i, 0, force_constant)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    return mol


def check_num_frags(mol, num_frags):
    if len(Chem.GetMolFrags(mol)) != num_frags:
        print(f'{Chem.MolToSmiles(mol)} has {len(Chem.GetMolFrags(mol))} frags')
        return False
    else:
        return True


# %%
if __name__ == '__main__':
    import copy
    import numpy as np
    import random
    import sys
    sys.path.append('/home/julius/soft/GB-GA/')
    from catalyst.utils import mols_from_smi_file, Timer
    from scoring_functions import calculate_scores_parallel
    from GB_GA import make_initial_population

    directory = '/home/julius/thesis/sims/ts_embed_scoring'
    numThreads = 1
    n_confs = 3
    # randomseed = 123 # das hat nen charegd cat
    randomseed = 101 # das hat nen fluo und das gibt probleme bei xyzmol und xyt opt
    # randomseed = 321
    generation = 0

    random.seed(randomseed)
    np.random.seed(randomseed)

    population = make_initial_population(5, '/home/julius/soft/GB-GA/ZINC_1000_amines.smi')

    from catalyst.utils import sdf2mol, draw3d, mol_from_xyz
    from catalyst.scoring import compare_angles, bonds_OK

    ts_dummy = sdf2mol('/home/julius/soft/GB-GA/catalyst/structures/ts_dummy.sdf')
    ts3d_structures = []
    for cat in population:
    # cat = population[0]

        ts2ds = connect_cat_2d(ts_dummy, cat)
        ts3d_energies = []
        print(Chem.MolToSmiles(cat))
        
        for i, ts2d in enumerate(ts2ds): # for each constitutional isomer
            test = ConstrainedEmbedMultipleConfsMultipleFrags(ts2d, ts_dummy, numConfs=1, force_constant=1e3)
            ts3d_structures.append(test)
            break
    # draw3d(test)
