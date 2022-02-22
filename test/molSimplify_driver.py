from my_utils.my_utils import cd
from my_utils.auto import shell


def create_cycleMS(new_core=None)


def main():

    # What ligand do i want to test?
    lig_smi = "C1=CC=CC=C1"
    core_file = '../templates/core_withHS.xyz'
    replig = 1
    suffix = "core_custom"
    run_dir = "Runs"

    core_str = f"molsimplify -core {core_file} -lig {lig_smi} -replig 1 -ligocc 3" \
               f" -ccatoms 24, 25, 26 -skipANN True -spin 1 -oxstate 3 -ffoption no" \
               f" -coord 5 -keepHs no -smicat 1"
    print(f"String passed to shell: {core_str}")
    out, err = shell(
        core_str,
        shell=False,
    )
    with open("job.out", "w") as f:
        f.write(out)
    with open("err.out", "w") as f:
        f.write(err)

    # Check for output structure

    # Pass  the output structure to cycle creation
    create_cycleMS()

if __name__ == '__main__':
    main()