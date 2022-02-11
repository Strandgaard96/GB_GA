# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import time
import json
import subprocess
import copy
import shutil
import os  # this is jsut vor OMP
import sys

sys.path.append("/home/julius/soft/GB-GA/")

from catalyst.utils import (
    sdf2mol,
    Timer,
    hartree2kcalmol,
)
from catalyst.make_structures import (
    ConstrainedEmbedMultipleConfsMultipleFrags,
    connect_cat_2d,
)
from catalyst.xtb_utils import xtb_optimize

reactant_dummy = sdf2mol("/home/julius/thesis/data/rr_dummy.sdf")
product_dummy = sdf2mol("/home/julius/thesis/data/pp_dummy.sdf")
frag_energies = np.sum(
    [-8.232710038092, -19.734652802142, -32.543971411432]
)  # 34 atoms
# %%


def path_scoring(individual, args_list):
    (
        n_confs,
        randomseed,
        timing_logger,
        warning_logger,
        directory,
        cpus_per_molecule,
    ) = args_list
    warning = None
    ind_dir = os.path.join(
        directory, f"G{individual.idx[0]:02d}_I{individual.idx[1]:02d}"
    )
    try:
        t1 = Timer(logger=None)
        t1.start()
        charge = Chem.GetFormalCharge(individual.rdkit_mol)
        reactant = make_reactant(
            cat=individual.rdkit_mol,
            ind_num=individual.idx[1],
            gen_num=individual.idx[0],
            n_confs=n_confs,
            randomseed=randomseed,
            numThreads=cpus_per_molecule,
            warning_logger=warning_logger,
            directory=directory,
        )
        cat_energy = calc_cat(
            reactant=reactant,
            ind_directory=ind_dir,
            cat_charge=charge,
            numThreads=cpus_per_molecule,
        )
        product_file = make_product(
            reactant=reactant,
            cat_charge=charge,
            cat=individual.rdkit_mol,
            ind_num=individual.idx[1],
            gen_num=individual.idx[0],
            numThreads=cpus_per_molecule,
            directory=directory,
        )
        path_dir = xtb_path(
            ind_num=individual.idx[1],
            gen_num=individual.idx[0],
            charge=charge,
            numThreads=cpus_per_molecule,
            inp_file="/home/julius/soft/GB-GA/catalyst/path_template.inp",
            directory=directory,
        )
        ts_energy = run_refinement(path_dir, charge)
        energy = hartree2kcalmol(ts_energy - cat_energy - frag_energies)
    except Exception as e:
        # if warning_logger:
        #     warning_logger.warning(f'{individual.smiles}: {traceback.print_exc()}')
        # else:
        # print(f'{individual.smiles}: {traceback.print_exc()}')
        energy = None
        warning = str(e)
    individual.energy = energy
    individual.warnings.append(warning)
    # shutil.rmtree(ind_dir)
    elapsed_time = t1.stop()
    individual.timing = elapsed_time
    return individual


def make_reactant(
    cat,
    ind_num,
    gen_num,
    n_confs=5,
    pruneRmsThresh=-1,
    randomseed=101,
    numThreads=1,
    warning_logger=None,
    directory=".",
):
    ind_dir = f"G{gen_num:02d}_I{ind_num:02d}"
    cat_charge = Chem.GetFormalCharge(cat)
    reactants2d = connect_cat_2d(reactant_dummy, cat)
    reactant3d_energies = []
    reactant3d_files = []
    for i, reactant2d in enumerate(reactants2d):  # for each constitutional isomer
        bonds_ok = False
        force_constant = 7500
        max_tries = 8
        tries = 0
        while not bonds_ok and tries < max_tries:
            reactant2d_copy = copy.deepcopy(reactant2d)
            reactant3d = ConstrainedEmbedMultipleConfsMultipleFrags(
                mol=reactant2d_copy,
                core=reactant_dummy,
                numConfs=int(n_confs),
                randomseed=int(randomseed),
                numThreads=int(numThreads),
                force_constant=int(force_constant),
                pruneRmsThresh=pruneRmsThresh,
                atoms2join=[(11, 29)],
            )
            if bonds_OK(reactant3d, threshold=2):
                bonds_ok = True
            else:
                force_constant = force_constant / 1.2
            tries += 1
            # if tries == max_tries:
            #     if warning_logger:
            #         warning_logger.warning(f'Embedding of {Chem.MolToSmiles(ts2d)} was not successful in {max_tries} tries with final forceconstant={force_constant}')
            #     raise Exception(f'Embedding was not successful in {max_tries} tries')
        # xTB optimite TS
        reactant3d_file, reactant3d_energy = xtb_optimize(
            reactant3d,
            method="gfnff",
            name=os.path.join(ind_dir, f"const_iso{i:03d}"),
            charge=cat_charge,
            constrains="/home/julius/thesis/data/constr.inp",
            scratchdir=directory,
            remove_tmp=False,
            return_file=True,
            numThreads=numThreads,
        )
        # here should be a num frag check
        reactant3d_energies.append(reactant3d_energy)
        reactant3d_files.append(reactant3d_file)
    reactant_energy = min(reactant3d_energies)
    min_e_index = reactant3d_energies.index(reactant_energy)
    reactant_file = reactant3d_files[
        min_e_index
    ]  # lowest energy Reactant constitutional isomer
    # make gfn2 opt
    gfn2_opt_dir = os.path.join(
        os.path.dirname(os.path.dirname(reactant_file)), "gfn2_opt_reactant"
    )
    os.mkdir(gfn2_opt_dir)
    minE_reactant_regioisomer_file = os.path.join(
        gfn2_opt_dir, "minE_reactant_isomer.xyz"
    )
    shutil.move(reactant_file, minE_reactant_regioisomer_file)
    reactant, gfn2_reactant_energy = xtb_optimize(
        minE_reactant_regioisomer_file,
        method="gfn2",
        constrains="/home/julius/thesis/data/constr.inp",
        name=None,
        charge=cat_charge,
        scratchdir=directory,
        remove_tmp=False,
        return_file=False,
        numThreads=numThreads,
    )
    return reactant


def bonds_OK(mol, threshold):
    mol_confs = mol.GetConformers()
    for conf in mol_confs:
        for bond in mol.GetBonds():
            length = AllChem.GetBondLength(
                conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            )
            if length > threshold:
                return False
    return True


def isolate_cat(mol, scarfold):
    if mol.HasSubstructMatch(scarfold):
        cat = AllChem.ReplaceCore(mol, scarfold, replaceDummies=False)
    else:
        raise Exception(
            f"Change in BO or connectivity of Catalyst in reactant minimization (isolate_cat): {Chem.MolToSmiles(mol)}"
        )
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


def calc_cat(reactant, ind_directory, cat_charge, numThreads):
    # cat_path = os.path.join(ind_directory, 'catalyst')
    cat = isolate_cat(reactant, reactant_dummy)
    cat_opt_file, cat_energy = xtb_optimize(
        cat,
        name="catalyst",
        method="gfn2",
        charge=cat_charge,
        scratchdir=ind_directory,
        remove_tmp=False,
        return_file=True,
        numThreads=numThreads,
    )
    return cat_energy


# from https://pschmidtke.github.io/blog/rdkit/3d-editor/2021/01/23/grafting-fragments.html
def getAttachmentVector(mol):
    """for a fragment to add, search for the position of the attachment point and extract the atom id's of the attachment point and the connected atom (currently only single bond supported)
    mol: fragment passed as rdkit molecule
    return: tuple (atom indices)
    """

    rindex = -1
    rindexNeighbor = -1
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            rindex = atom.GetIdx()
            neighbours = atom.GetNeighbors()
            if len(neighbours) == 1:
                rindexNeighbor = neighbours[0].GetIdx()
            else:
                print("two attachment points not supported yet")
                return None
    return (rindex, rindexNeighbor)


def connectMols(mol1, mol2, atom1, atom2):
    """function copied from here https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py"""
    combined = Chem.CombineMols(mol1, mol2)
    emol = Chem.EditableMol(combined)
    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
    atom1_idx = atom1.GetIdx()
    atom2_idx = atom2.GetIdx()
    bond_order = atom2.GetBonds()[0].GetBondType()
    emol.AddBond(neighbor1_idx, neighbor2_idx + mol1.GetNumAtoms(), order=bond_order)
    emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
    emol.RemoveAtom(atom1_idx)
    mol = emol.GetMol()
    return mol


def get_connection_id(mol):
    for atom in mol.GetAtomWithIdx(0).GetNeighbors():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
            connection_id = atom.GetIdx()
            break
    return connection_id


def make_product(
    reactant, cat_charge, cat, ind_num, gen_num, numThreads=1, directory="."
):
    # cut cat from reactant and attach to product_dummy
    bs = [reactant.GetBondBetweenAtoms(0, 1).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(
        reactant, bs, addDummies=True, dummyLabels=[(1, 1)]
    )
    fragments = AllChem.GetMolFrags(fragments_mol, asMols=True)

    for fragment in fragments:
        if fragment.HasSubstructMatch(cat):
            catalyst = fragment
            break

    try:
        catalyst
    except NameError:
        raise Exception(
            f"Change in BO or connectivity of Catalyst in reactant minimization (make_product): {Chem.MolToSmiles(fragments[0])} != {Chem.MolToSmiles(cat)}"
        )

    cat_dummyidx = getAttachmentVector(catalyst)  # R N
    cat_overlap = cat_dummyidx + (get_connection_id(catalyst),)
    product_dummyidx = getAttachmentVector(product_dummy)  # R C
    product_overlap = product_dummyidx + (1,)

    rms = AllChem.AlignMol(
        catalyst,
        product_dummy,
        atomMap=(
            (cat_overlap[0], product_overlap[-1]),
            (cat_overlap[1], product_overlap[-2]),
            (cat_overlap[2], product_overlap[-3]),
        ),
    )

    bs = [catalyst.GetBondBetweenAtoms(get_connection_id(catalyst), 0).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(
        catalyst, bs, addDummies=True, dummyLabels=[(1, 1)]
    )
    fragments = AllChem.GetMolFrags(fragments_mol, asMols=True)

    for fragment in fragments:
        if fragment.HasSubstructMatch(cat):
            catalyst2 = fragment
            break
    cat2_dummyidx = getAttachmentVector(catalyst2)
    product = connectMols(
        product_dummy,
        catalyst2,
        product_dummy.GetAtomWithIdx(product_dummyidx[0]),
        catalyst2.GetAtomWithIdx(cat2_dummyidx[0]),
    )
    flags = Chem.SanitizeMol(product)

    product_opt, gfn2_reactant_energy = xtb_optimize(
        product,
        method="gfn2",
        constrains="/home/julius/thesis/data/constr.inp",
        name=f"G{gen_num:02d}_I{ind_num:02d}/gfn2_opt_product",
        charge=cat_charge,
        scratchdir=directory,
        remove_tmp=False,
        return_file=False,
        numThreads=numThreads,
    )

    constrained_atoms = []
    for atom in product_opt.GetAtoms():
        if atom.GetIdx() < 34:  # get_connection_id(product_opt):
            constrained_atoms.append(atom.GetIdx())
        else:
            break

    # AllChem.AlignMol(reactant, product_opt, atomMap=list(
    #     zip(constrained_atoms, constrained_atoms)))  # atom 11 is included but whatever

    number_of_atoms = product_opt.GetNumAtoms()
    symbols = [a.GetSymbol() for a in product_opt.GetAtoms()]
    conf = product_opt.GetConformers()[0]
    aligned_file = os.path.join(
        directory, f"G{gen_num:02d}_I{ind_num:02d}/gfn2_opt_product/aligned_product.xyz"
    )
    with open(aligned_file, "w") as _file:
        _file.write(str(number_of_atoms) + "\n")
        _file.write(f"{Chem.MolToSmiles(fragment)}\n")
        for atom, symbol in enumerate(symbols):
            p = conf.GetAtomPosition(atom)
            line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
            _file.write(line)

    return aligned_file


def extract_xtb_structures(path_file):
    dest_mol_dir = os.path.dirname(os.path.dirname(path_file))
    xtb_sp_dir = os.path.join(dest_mol_dir, "sp_refinement")
    try:
        os.makedirs(xtb_sp_dir)
        with open(path_file, "r") as _file:
            count = 0
            index = 1
            sp_file = None
            for line in _file:
                if count == 0:
                    n_lines = int(line.split()[0]) + 2
                if count % n_lines == 0:
                    if sp_file:
                        sp_file.close()
                    sp_filename = "sp_{}.xyz".format(index)
                    sp_file = open(os.path.join(xtb_sp_dir, sp_filename), "w")
                    index += 1
                sp_file.write(line)
                count += 1
            if sp_file:
                sp_file.close()
    except:
        print(f"{xtb_sp_dir} already exists.")
    return xtb_sp_dir


def xtb_path(
    ind_num,
    gen_num,
    charge,
    numThreads,
    inp_file="/home/julius/soft/GB-GA/catalyst/path_template.inp",
    directory=".",
):
    ind_dir = os.path.join(directory, f"G{gen_num:02d}_I{ind_num:02d}")
    path_dir = os.path.join(ind_dir, "path")
    os.mkdir(path_dir)
    shutil.copyfile(
        os.path.join(ind_dir, "gfn2_opt_reactant/xtbopt.xyz"),
        os.path.join(path_dir, "reactant.xyz"),
    )
    shutil.copyfile(
        os.path.join(ind_dir, "gfn2_opt_product/aligned_product.xyz"),
        os.path.join(path_dir, "product.xyz"),
    )
    os.environ["OMP_NUM_THREADS"] = f"{numThreads},1"
    os.environ["MKL_NUM_THREADS"] = f"{numThreads}"
    os.environ["OMP_STACKSIZE"] = "6G"
    p = subprocess.Popen(
        f"/home/julius/soft/xtb-6.3.3/bin/xtb reactant.xyz --path product.xyz --input {inp_file} --gfn2 --chrg {charge} --alpb methanol --verbose > xtb_path.out",
        shell=True,
        cwd=path_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    while p.poll() == None:
        time.sleep(30)
        if "xtbpath_ts.xyz" in os.listdir(path_dir):
            time.sleep(2)  # so that xtbpath_0.xyz is completly written
            return os.path.join(path_dir)
        if "xtbpath_3.xyz" in os.listdir(path_dir):
            raise Exception(f"xTB path did not yield product in 3 runs")
        p.poll()
    (results, errors) = p.communicate()
    if errors == "":
        print(results)
    else:
        raise Exception(errors)


def run_refinement(path_dir, charge):
    path_file = os.path.join(path_dir, "xtbpath_0.xyz")
    if not os.path.exists(path_file):
        raise Exception(f"no xTB path file found")
    sp_dir = extract_xtb_structures(path_file)
    run_sps(sp_dir, charge)
    inter_dir = find_xtb_max_from_sp_interpolation(sp_dir, True)
    atom_numbers_list, coordinates_list, n_atoms = get_coordinates(inter_dir)
    make_sp_interpolation(inter_dir, atom_numbers_list, coordinates_list, n_atoms, 15)
    run_sps(inter_dir, charge)
    refined_path_ts_energy = find_xtb_max_from_sp_interpolation(inter_dir, False)
    return refined_path_ts_energy


def run_sps(sp_dir, charge):
    for _file in os.listdir(sp_dir):
        if _file.endswith(".xyz"):
            p = subprocess.Popen(
                f'/home/julius/soft/xtb-6.3.3/bin/xtb {_file} --gfn2 --chrg {charge} --json --alpb methanol > {_file.split(".")[0]}.out',
                shell=True,
                cwd=sp_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output, err = p.communicate()
            shutil.move(
                os.path.join(sp_dir, "xtbout.json"),
                os.path.join(sp_dir, f'{_file.split(".")[0]}.json'),
            )


def find_xtb_max_from_sp_interpolation(sp_directory, extract_max_structures):
    """
    when sp calculations are finished: find the structure with maximum xtb
    energy
    """
    energies = []
    path_points = []
    inter_dir = ""

    files = [
        os.path.join(sp_directory, f)
        for f in os.listdir(sp_directory)
        if f.endswith("json")
    ]
    files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    for file_name in files:
        path_point = int(os.path.basename(file_name).split(".")[0].split("_")[-1])
        path_points.append(path_point)
        with open(file_name, "r") as _file:
            data = json.load(_file)
            energy_au = data["total energy"]
        energies.append(energy_au)
    energies_kcal = np.array(energies) * 627.509
    energies_kcal = energies_kcal - energies_kcal[0]
    max_index = energies.index(max(energies))
    if extract_max_structures:
        dest_mol_dir = os.path.dirname(sp_directory)
        inter_dir = os.path.join(dest_mol_dir, "interpolation")
        if os.path.exists(inter_dir):
            shutil.rmtree(inter_dir)
        os.mkdir(inter_dir)
        max_point = path_points[max_index]
        if max_index == 0:
            raise Exception("Reactant is max Energy structure along the path")
        shutil.copy(
            os.path.join(sp_directory, "sp_" + str(max_point - 1) + ".xyz"),
            os.path.join(inter_dir, "max_structure-1.xyz"),
        )
        shutil.copy(
            os.path.join(sp_directory, "sp_" + str(max_point) + ".xyz"),
            os.path.join(inter_dir, "max_structure.xyz"),
        )
        shutil.copy(
            os.path.join(sp_directory, "sp_" + str(max_point + 1) + ".xyz"),
            os.path.join(inter_dir, "max_structure+1.xyz"),
        )
        return inter_dir
    else:
        max_point = path_points[max_index]
        try:
            shutil.copy(
                os.path.join(sp_directory, "path_point_" + str(max_point) + ".xyz"),
                os.path.join(os.path.dirname(sp_directory), "refined_TS.xyz"),
            )
        except:
            pass
        return max(energies)


def get_coordinates(interpolation_dir):
    """
    Extrapolate around maximum structure on the xtb surface to make DFT single
    point calculations in order to choose the best starting point for TS
    optimization. Should return this starting point structure
    """
    structure_files_list = os.listdir(interpolation_dir)
    n_structures = len(structure_files_list)
    atom_numbers_list = []
    coordinates_list = []
    for i in range(n_structures):
        atom_numbers = []
        with open(
            os.path.join(interpolation_dir, structure_files_list[i]), "r"
        ) as struc_file:
            line = struc_file.readline()
            n_atoms = int(line.split()[0])
            struc_file.readline()
            coordinates = np.zeros((n_atoms, 3))
            for j in range(n_atoms):
                line = struc_file.readline().split()
                atom_number = line[0]
                atom_numbers.append(atom_number)
                coordinates[j, :] = np.array([np.float(num) for num in line[1:]])
        atom_numbers_list.append(atom_numbers)
        coordinates_list.append(coordinates)
    return atom_numbers_list, coordinates_list, n_atoms


def make_sp_interpolation(
    interpolation_dir, atom_numbers_list, coordinates_list, n_atoms, n_points
):
    """
    From the given structures in coordinates_list xyz files are created by
    extrapolating between those structures with n_points between each structure
    creates a directory "path" with those .xyz files
    """
    max_structure_files = [
        "max_structure-1.xyz",
        "max_structure.xyz",
        "max_structure+1.xyz",
    ]

    for _file in max_structure_files:
        os.remove(os.path.join(interpolation_dir, _file))
    n_structures = len(coordinates_list)
    with open(os.path.join(interpolation_dir, "path_file.txt"), "w") as path_file:
        for i in range(n_structures - 1):
            difference_mat = coordinates_list[i + 1] - coordinates_list[i]
            for j in range(n_points + 1):
                path_xyz = coordinates_list[i] + j / n_points * difference_mat
                path_xyz = np.matrix(path_xyz)
                file_path = os.path.join(
                    interpolation_dir, "path_point_" + str(i * n_points + j) + ".xyz"
                )
                with open(file_path, "w+") as _file:
                    _file.write(str(n_atoms) + "\n\n")
                    path_file.write(str(n_atoms) + "\n\n")
                    for atom_number, line in zip(atom_numbers_list[i], path_xyz):
                        _file.write(atom_number + " ")
                        path_file.write(atom_number + " ")
                        np.savetxt(_file, line, fmt="%.6f")
                        np.savetxt(path_file, line, fmt="%.6f")


# %%
if __name__ == "__main__":
    import pickle
    import sys

    sys.path.append("/home/julius/soft/GB-GA/")

    ind_file = sys.argv[-7]
    nconfs = sys.argv[-6]
    randomseed = sys.argv[-5]
    timing_logger = sys.argv[-4]
    warning_logger = sys.argv[-3]
    directory = sys.argv[-2]
    cpus_per_molecule = sys.argv[-1]

    args_list = [
        nconfs,
        randomseed,
        timing_logger,
        warning_logger,
        directory,
        cpus_per_molecule,
    ]
    with open(ind_file, "rb") as f:
        ind = pickle.load(f)

    individual = path_scoring(ind, args_list)

    with open(ind_file, "wb+") as new_file:
        pickle.dump(individual, new_file)

# %%
