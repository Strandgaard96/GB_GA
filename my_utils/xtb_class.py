import concurrent.futures
import os
import random
import shutil
import string
from pathlib import Path

import numpy as np
from rdkit import Chem

from my_utils.classes import core
from my_utils.xtb_utils import check_bonds, run_xtb


class XTB_optimizer:
    """Base XTB optimizer class"""

    def __init__(self):

        # Initialize default xtb values
        self.method = "ff"
        self.workers = 1
        # Xtb runner function
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
    def __init__(self, mol, scoring_options, **kwargs):

        super().__init__(**kwargs)
        self.mol = mol
        self.options = scoring_options

        # Set xtb options
        self.add_options_to_cmd(self.options)

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
        # Locate atoms to contrain
        match = (
            np.array(molecule.GetSubstructMatch(core)) + 1
        )  # indexing starts with 0 for RDKit but 1 for xTB
        match = sorted(match)
        assert len(match) == core.GetNumAtoms(), "ERROR! Complete match not found."

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
            idxs = []
            for elem in molecule.GetAtoms():
                idxs.append(elem.GetIdx() + 1)
            match = [idx for idx in idxs if idx not in match]

        for elem in path:
            # Write the xcontrol file
            with open(os.path.join(elem, "xcontrol.inp"), "w") as f:
                f.write("$fix\n")
                f.write(f' atoms: {",".join(map(str, match))}\n')
                f.write("$end\n")
        return

    @staticmethod
    def copy_logfile(conf_paths, name="opt.log"):
        for elem in conf_paths:
            shutil.copy(os.path.join(elem, "xtbopt.log"), os.path.join(elem, name))

    def optimize_schrock(self):

        # Check mol
        n_confs = self._check_mol(self.mol)

        # Write input files
        xyz_files, conf_paths = self._write_xtb_input_files(
            self.mol, "xtbmol", destination=self.name
        )

        # Make input constrain file
        self._make_input_constrain_file(
            self.mol, core=core, path=conf_paths, NH3=True, N2=True
        )

        workers = np.min([self.options["cpus_per_task"], self.options["n_confs"]])
        cpus_per_worker = self.options["cpus_per_task"] // workers
        # cpus_per_worker = 1
        print(f"workers: {workers}, cpus_per_worker: {cpus_per_worker}")
        args = [
            (xyz_file, self.cmd, cpus_per_worker, conf_paths[i], "ff")
            for i, xyz_file in enumerate(xyz_files)
        ]
        result = self.optimize(args)

        # Store the log file
        self.copy_logfile(conf_paths, name="ffopt.log")

        # Run non_constrained gfn2
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
        result = self.optimize(args)

        # Store the log file
        self.copy_logfile(conf_paths, name="constrained_opt.log")

        # Constrain only N reactants and Mo
        self._make_input_constrain_file(
            self.mol,
            core=Chem.MolFromSmiles("[Mo]"),
            path=conf_paths,
            NH3=True,
            N2=True,
        )
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

            self._make_input_constrain_file(
                self.mol,
                core=Chem.MolFromSmiles("[Mo]"),
                path=conf_paths,
                NH3=True,
                N2=True,
                Mo_bond=True,
            )
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

        print("Finished all optimizations")
        energies, geometries, minidx = self._get_results(result)

        # Clean up
        if self.options["cleanup"]:
            shutil.rmtree(self.name)

        final_geom, bond_change = check_bonds(
            self.options.get("bare", False),
            self.options["charge"],
            conf_paths,
            geometries,
            minidx,
            self.mol,
            self.options.get("print_xyz", True),
        )
        if bond_change:
            return 9999, None, None
        else:
            return energies[minidx], final_geom, minidx.item()

    def _get_results(self, result):
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
    def _check_mol(mol):
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
