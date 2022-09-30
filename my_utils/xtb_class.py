import concurrent.futures
import copy
import os
import random
import shutil
import string
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Geometry import Point3D

from my_utils.xtb_utils import check_bonds, run_xtb

file = "templates/core_noHS.mol"
core = Chem.MolFromMolFile(file, removeHs=False, sanitize=False)
"""Mol: 
mol object of the Mo core with dummy atoms instead of ligands
"""


class XTB_optimizer:
    """Base XTB optimizer class"""

    def __init__(self):

        # Initialize default xtb values
        self.method = "ff"
        self.workers = 1
        # xtb runner function
        self.xtb_runner = run_xtb
        # xtb options
        self.XTB_OPTIONS = {
            "opt": "tight",
        }

        # Start with ff optimization
        cmd = f"xtb --gfn{self.method}"
        for key, value in self.XTB_OPTIONS.items():
            cmd += f" --{key} {value}"
        self.cmd = cmd

    def add_options_to_cmd(self, option_dict):
        """From passed dict get xtb options if it has the appropriate
        keys and add to xtb string command"""

        # XTB options to check for
        options = ["gbsa", "spin", "charge", "uhf", "input", "opt"]

        # Get commandline options
        commands = {k: v for k, v in option_dict.items() if k in options}
        for key, value in commands.items():
            self.cmd += f" --{key} {value}"

    def optimize(self, args):
        """Do paralell optimization of all the entries in args"""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers
        ) as executor:
            results = executor.map(self.xtb_runner, args)
        return results

    @staticmethod
    def _write_xtb_input_files(fragment, name, destination="."):
        """Utility method to write xyz input files from mol object"""

        number_of_atoms = fragment.GetNumAtoms()
        symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
        conformers = fragment.GetConformers()
        file_paths = []
        conf_paths = []
        for i, conf in enumerate(conformers):
            conf_path = os.path.join(destination, f"conf{i:03d}")
            conf_paths.append(conf_path)

            if os.path.exists(conf_path):
                shutil.rmtree(conf_path)
            os.makedirs(conf_path)

            file_name = f"{name}{i:03d}.xyz"
            file_path = os.path.join(conf_path, file_name)
            with open(file_path, "w") as _file:
                _file.write(str(number_of_atoms) + "\n")
                _file.write(f"{Chem.MolToSmiles(fragment)}\n")
                for atom, symbol in enumerate(symbols):
                    p = conf.GetAtomPosition(atom)
                    line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                    _file.write(line)
            file_paths.append(file_path)
        return file_paths, conf_paths


class XTB_optimize_schrock(XTB_optimizer):
    """Specific xtb optimizer class for the schrock intermediates"""

    def __init__(self, mol, scoring_options, **kwargs):
        """

        Args:
            mol (Chem.rdchem.Mol): Mol object to score
            scoring_options (dict): Scoring options for xtb
        """

        # Inherit the basic xtb functionality from XTB_OPTIMIZER
        super().__init__(**kwargs)

        # Set class attributes
        self.mol = mol
        self.options = scoring_options

        # Set additional xtb options
        self.add_options_to_cmd(self.options)

        # Set folder name if given options dict
        if not "name" in self.options:
            self.name = "tmp_" + "".join(
                random.choices(string.ascii_uppercase + string.digits, k=4)
            )
        else:
            self.name = self.options["name"]

        # set SCRATCH if environmental variable
        try:
            self.scr_dir = os.environ["SCRATCH"]
        except:
            self.scr_dir = os.getcwd()
        print(f"SCRATCH DIR = {self.scr_dir}")

    @staticmethod
    def _make_input_constrain_file(
        molecule, core, path, NH3=False, N2=False, Mo_bond=False
    ):
        """Make input constrain file

        Args:
            molecule (Chem.rdchem.Mol): molecule to match on
            core (Chem.rdchem.Mol): mol object specifying what to constrain
            path (Path): Path to various conformers
            NH3 (bool): Constrain NH3 on the core
            N2 (bool): Constrain N2 on the core
            Mo_bond (bool): Constrain the Mo-N bond
        """

        # Locate core atoms
        match = (
            np.array(molecule.GetSubstructMatch(core)) + 1
        )  # indexing starts with 0 for RDKit but 1 for xTB
        match = sorted(match)
        assert len(match) == core.GetNumAtoms(), "ERROR! Complete match not found."

        # See if match list should be extended.
        if NH3:
            NH3_match = Chem.MolFromSmarts("[NH3]")
            NH3_match = Chem.AddHs(NH3_match)
            NH3_sub_match = np.array(molecule.GetSubstructMatch(NH3_match)) + 1
            match.extend(NH3_sub_match)
        if N2:
            N2_match = Chem.MolFromSmarts("N#N")
            N2_sub_match = np.array(molecule.GetSubstructMatch(N2_match)) + 1
            match.extend(N2_sub_match)
        if Mo_bond:
            # Constrain everything that is not spcified in core.
            # If N2 or NH3 flag was also set, these are also not contrained
            idxs = []
            for elem in molecule.GetAtoms():
                idxs.append(elem.GetIdx() + 1)
            match = [idx for idx in idxs if idx not in match]

        # Loop conformer paths
        for elem in path:
            # Write the xcontrol file
            with open(os.path.join(elem, "xcontrol.inp"), "w") as f:
                f.write("$fix\n")
                f.write(f' atoms: {",".join(map(str, match))}\n')
                f.write("$end\n")
        return

    @staticmethod
    def _constrain_N(molecule, path, NH3=False, N2=False):
        """Make input constrain file

        Args:
            molecule (Chem.rdchem.Mol): molecule to match on
            core (Chem.rdchem.Mol): mol object specifying what to constrain
            path (Path): Path to various conformers
            NH3 (bool): Constrain NH3 on the core
            N2 (bool): Constrain N2 on the core
            Mo_bond (bool): Constrain the Mo-N bond
        """
        # See if match list should be extended.

        match = []
        if NH3:
            NH3_match = Chem.MolFromSmarts("[NH3]")
            NH3_match = Chem.AddHs(NH3_match)
            NH3_sub_match = np.array(molecule.GetSubstructMatch(NH3_match)) + 1
            match.extend(NH3_sub_match)
        if N2:
            N2_match = Chem.MolFromSmarts("N#N")
            N2_sub_match = np.array(molecule.GetSubstructMatch(N2_match)) + 1
            match.extend(N2_sub_match)

        # Loop conformer paths
        for elem in path:
            # Write the xcontrol file
            with open(os.path.join(elem, "xcontrol.inp"), "w") as f:
                f.write("$constrain\n")
                f.write(f' atoms: {",".join(map(str, match))}\n')
                f.write("$end\n")
        return

    @staticmethod
    def copy_logfile(conf_paths, name="opt.log"):
        """Copy xtbopt.log file to new name"""
        for elem in conf_paths:
            shutil.copy(os.path.join(elem, "xtbopt.log"), os.path.join(elem, name))

    def optimize_schrock(self):
        """Optimize the given mol object

        Returns:
            mol_opt: optimized mol object with all the conformers
        """

        # Check mol
        n_confs = self._check_mol(self.mol)

        # Write input files
        xyz_files, conf_paths = self._write_xtb_input_files(
            self.mol, "xtbmol", destination=self.name
        )

        # Make input constrain file. Constrain only to Mo core atoms
        self._make_input_constrain_file(
            self.mol, core=core, path=conf_paths, NH3=True, N2=True
        )

        # Set paralellization options
        self.workers = np.min([self.options["cpus_per_task"], self.options["n_confs"]])
        cpus_per_worker = self.options["cpus_per_task"] // self.workers
        print(f"workers: {self.workers}, cpus_per_worker: {cpus_per_worker}")

        # Create args tuple and submit ff calculation
        args = [
            (xyz_file, self.cmd, cpus_per_worker, conf_paths[i], "ff")
            for i, xyz_file in enumerate(xyz_files)
        ]
        result = self.optimize(args)

        # Store the log file under given name
        self.copy_logfile(conf_paths, name="ffopt.log")

        # Change from ff to given method
        self.cmd = self.cmd.replace("gfnff", f"gfn {self.options['method']}")

        # Get the new input files and args
        xyz_files = [Path(xyz_file).parent / "xtbopt.xyz" for xyz_file in xyz_files]
        args = [
            (
                xyz_file,
                self.cmd,
                cpus_per_worker,
                conf_paths[i],
                f"const_gfn{self.options['method']}",
            )
            for i, xyz_file in enumerate(xyz_files)
        ]

        # Optimize with current input constrain file. Only the Mo core.
        result = self.optimize(args)

        # Store the log file
        self.copy_logfile(conf_paths, name="constrained_opt.log")

        # Constrain only N reactants and Mo. Let the rest of the core optimize
        self._make_input_constrain_file(
            self.mol,
            core=Chem.MolFromSmiles("[Mo]"),
            path=conf_paths,
            NH3=True,
            N2=True,
        )

        # Get new args and optimize
        args = [
            (
                xyz_file,
                self.cmd,
                cpus_per_worker,
                conf_paths[i],
                f"gfn{self.options['method']}",
            )
            for i, xyz_file in enumerate(xyz_files)
        ]
        result = self.optimize(args)

        self.copy_logfile(conf_paths, name="Mo_gascon.log")

        if self.options.get("bond_opt", False):

            # Get constrain file for only Mo plus NH3 or N2 atoms.
            self._make_input_constrain_file(
                self.mol,
                core=Chem.MolFromSmiles("[Mo]"),
                path=conf_paths,
                NH3=True,
                N2=True,
                Mo_bond=True,
            )

            # Optimize the Mo-N* bond
            args = [
                (
                    xyz_file,
                    self.cmd,
                    cpus_per_worker,
                    conf_paths[i],
                    f"gfn{self.options['method']}",
                )
                for i, xyz_file in enumerate(xyz_files)
            ]

            result = self.optimize(args)

        if self.options.get("full_relax", False):
            self._constrain_N(self.mol, path=conf_paths, NH3=True, N2=True)
            # Optimize the Mo-N* bond
            args = [
                (
                    xyz_file,
                    self.cmd,
                    cpus_per_worker,
                    conf_paths[i],
                    f"full_relax",
                )
                for i, xyz_file in enumerate(xyz_files)
            ]

            result = self.optimize(args)

        print("Finished all optimizations")
        # Add optimized conformers to mol_opt
        mol_opt = copy.deepcopy(self.mol)
        n_confs = mol_opt.GetNumConformers()

        # Check if any organic bonds have formed/broken
        bond_changes = check_bonds(
            mol_opt,
            conf_paths,
            charge=self.options["charge"],
        )

        mol_opt.RemoveAllConformers()

        energies = []
        # Add optimized conformers
        for i, res in enumerate(result):
            if res:
                if not bond_changes[i] and res["energy"]:
                    energies.append(res["energy"])
                    self._add_conformer2mol(
                        mol=mol_opt,
                        atoms=res["atoms"],
                        coords=res["coords"],
                        energy=res["energy"],
                        bond_change=bond_changes[i],
                    )
            else:
                print(f"Conformer {i} did not converge.")

        # Reset confIDs (starting from 0)
        confs = mol_opt.GetConformers()
        print(len(confs))
        for i in range(len(confs)):
            confs[i].SetId(i)

        # if "constrain_atoms" in options and len(options["constrain_atoms"]) > 0:
        #    _ = AllChem.AlignMolConformers(mol_opt, atomIds=options["constrain_atoms"])

        # Clean up
        if self.options["cleanup"]:
            shutil.rmtree(self.name)

        return mol_opt, np.array(energies)

    def _get_results(self, result):
        """Parse xtb output tuple"""
        energies = []
        geometries = []
        for e, g in result:
            energies.append(e)
            geometries.append(g)
        arr = np.array(energies, dtype=np.float)
        try:
            minidx = np.nanargmin(arr)
            return energies, geometries, minidx
        except ValueError:
            print(
                "All-Nan slice encountered, setting minidx to None and returning 9999 energy"
            )
            energies = 9999
            geometries = None
            minidx = None
            return energies, geometries, minidx

    @staticmethod
    def _add_conformer2mol(mol, atoms, coords, energy=None, bond_change=None):
        """Add Conformer to rdkit.mol object."""
        conf = Chem.Conformer()
        for i in range(mol.GetNumAtoms()):
            # assert that same atom type
            assert (
                mol.GetAtomWithIdx(i).GetSymbol() == atoms[i]
            ), "Order of atoms if not the same in CREST output and rdkit Mol"
            x, y, z = coords[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

    @staticmethod
    def _check_mol(mol):
        """Check for implicit hydrogens and get number of conformers"""
        assert isinstance(mol, Chem.rdchem.Mol)
        if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
            raise Exception("Implicit Hydrogens")
        conformers = mol.GetConformers()
        n_confs = len(conformers)
        if not conformers:
            raise Exception("Mol is not embedded")
        elif not conformers[-1].Is3D():
            raise Exception("Conformer is not 3D")
        return n_confs
