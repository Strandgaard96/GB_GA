import py3Dmol
from rdkit import Chem

def draw_mol_with_highlights(mol, hit_ats, style=None):
    """Draw molecule in 3D with highlighted atoms. 

    Parameters
    ----------
    mol : RDKit molecule
    hit_ats : tuple of tuples
        atoms to highlight, from RDKit's GetSubstructMatches
    style : dict, optional
        drawing style, see https://3dmol.csb.pitt.edu/doc/$3Dmol.GLViewer.html for some examples

    Returns
    -------
    py3Dmol viewer
    """
    v = py3Dmol.view()
    if style is None: 
        style = {'stick':{'colorscheme':'grayCarbon', "linewidth": 0.1}}
    v.addModel(Chem.MolToMolBlock(mol), "mol") 
    v.setStyle({'model':0},style)
    hit_ats = [x for tup in hit_ats for x in tup]
    for atom in hit_ats:
        p = mol.GetConformer().GetAtomPosition(atom)
        v.addSphere({"center":{"x":p.x,"y":p.y,"z":p.z},"radius":0.9,"color":'green', "alpha": 0.8})
    v.setBackgroundColor('white')    
    v.zoomTo()
    return v

#### Example, use in Jupyter notebook: 
# from rdkit.Chem import AllChem
# cyclosporine_smiles = "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C"
# cyclosporine = Chem.AddHs(Chem.MolFromSmiles(cyclosporine_smiles))
# AllChem.EmbedMolecule(cyclosporine)

# patt = Chem.MolFromSmarts('O[H]')
# hit_ats = cyclosporine.GetSubstructMatches(patt)
# draw_mol_with_highlights(cyclosporine, hit_ats)

