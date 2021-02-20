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

def ConstrainedEmbedMultipleConfs(mol, core, numConfs=10, useTethers=True, coreConfId=-1, randomseed=2342,
                     getForceField=AllChem.UFFGetMoleculeForceField, numThreads=1):

    force_constant = 1e6
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