#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=magstr@dtu.dk  # The default value is the submitting user.
#SBATCH --partition=xeon40
#SBATCH -N 1      # Minimum of 2 nodes
#SBATCH -n 40     # 24 MPI processes per node, 48 tasks in total, appropriate for xeon24 nodes
#SBATCH --time=2-02:00:00
#SBATCH --mem=350G

# Initialize variable
job=orca

#Start ORCA job. ORCA is started using full pathname (necessary for parallel execution). Output file is written directly to submit directory on frontnode.
orca $job.inp > $job.out