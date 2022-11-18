import concurrent.futures
import os
import sys
from pathlib import Path

from support_mvp.auto import cd, shell


def sub_xtb(path, numThreads):

    os.environ["OMP_NUM_THREADS"] = f"{numThreads}"
    os.environ["MKL_NUM_THREADS"] = f"{numThreads}"
    os.environ["OMP_STACKSIZE"] = "1G"

    print(path)
    with cd(path.parent):

        cmd = "xtb struct.xyz --opt --input xcontrol.inp"
        shell((cmd, "test"))


def main():

    n_cores = int(sys.argv[1])

    cpus_per_worker = 2

    workers = n_cores // cpus_per_worker

    p = Path("/home/magstr/Documents/GB_GA/molS_drivers/dft_folder_11_15")
    paths = sorted(p.rglob("struct.xyz"))
    print(f"cores: {n_cores} workers : {workers}]")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(sub_xtb, paths, [2 for k in paths])


if __name__ == "__main__":
    main()
