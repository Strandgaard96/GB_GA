from my_utils.my_utils import cd
from my_utils.auto import shell
from pathlib import Path
import json
import shutil
import os

def create_cycleMS(new_core=None, smi_path=None):


    with open(smi_path,encoding='utf8') as f:
        smi_dict = json.load(f)

    for key,value in smi_dict.items():

        intermediate_cmd = f"molsimplify -core {new_core} -lig {value} -ligocc 1" \
                   f" -ccatoms 8 -skipANN True -spin 1 -oxstate 3 -ffoption no" \
                   f" -smicat 1 -suff {key} -rundir Runs_cycle"

        # Modify string if multiple ligands.
        # Which is the case for equatorial NN at Mo_N2_NH3.
        if key == "Mo_N2_NH3":
            intermediate_cmd = None
            intermediate_cmd = f"molsimplify -core {new_core} -oxstate 3 -lig [NH3],N#N" \
                               f" -ligocc 1,1 -ligloc True -ccatoms 8,8 -spin 1 -oxstate 3" \
                               f" -skipANN True -smicat [1],[1]" \
                               f" -ffoption no -suff {key} -rundir Runs_cycle"

        print(f"String passed to shell: {intermediate_cmd}")
        out, err = shell(
            intermediate_cmd,
            shell=False,
        )
        with open(f"job_{key}.out", "w") as f:
            f.write(out)
        with open(f"err_{key}.out", "w") as f:
            f.write(err)

    return

def main():

    # What ligand do i want to test?
    lig_smi = "C1=CC=CC=C1"
    core_file = Path('../templates/core_withHS.xyz')
    replig = 1
    suffix = "core_custom"
    run_dir = Path("Runs")

    core_cmd = f"molsimplify -core {core_file} -lig {lig_smi} -replig 1 -ligocc 3" \
               f" -ccatoms 24, 25, 26 -skipANN True -spin 1 -oxstate 3 -ffoption no" \
               f" -coord 5 -keepHs no -smicat 1"
    print(f"String passed to shell: {core_cmd}")
    out, err = shell(
        core_cmd,
        shell=False,
    )
    with open(f"job_{lig_smi}.out", "w") as f:
        f.write(out)
    with open(f"err_{lig_smi}.out", "w") as f:
        f.write(err)

    # Check for output structure
    new_core = sorted(run_dir.glob('**/*.xyz'))
    intermediate_smi_path = Path("../data/intermediate_smiles.json")

    # Move output structure to current dir
    shutil.copy(new_core[0], './new_core.xyz')

    if new_core:
        # Pass the output structure to cycle creation
        create_cycleMS(new_core='new_core.xyz', smi_path=intermediate_smi_path)

if __name__ == '__main__':
    main()