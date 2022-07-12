#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=orca
#SBATCH --mail-user=magstr@dtu.dk  # The default value is the submitting user.
#SBATCH --partition=xeon40
#SBATCH -N 1      # Minimum of 2 nodes
#SBATCH -n 40     # 24 MPI processes per node, 48 tasks in total, appropriate for xeon24 nodes
#SBATCH --time=2-02:00:00
#SBATCH --mem=350G

# Loading necessary modules
module load ORCA

# Initialize variable
job=orca

# Creating local scratch folder for the user on the computing node. 
# Set the scratchlocation variable to the location of the local scratch, e.g. /scratch or /localscratch
export scratchlocation=/scratch/magstr
tdir=$(mktemp -d $scratchlocation/orcajob__$SLURM_JOB_ID-XXXX)

# Copy only the necessary stuff in submit directory to scratch directory. Add more here if needed.

cp  $job.inp $tdir/
cp  *.xyz $tdir/

# Copy job and node info to beginning of outputfile
echo "Job execution start: $(date)" >  $job.out
echo "Slurm Job ID is: ${SLURM_JOB_ID}" >  $job.out
echo "Slurm Job name is: ${SLURM_JOB_NAME}" >  $job.out

#Start ORCA job. ORCA is started using full pathname (necessary for parallel execution). Output file is written directly to submit directory on frontnode.
/home/modules/software/ORCA/5.0.1-gompi-2021a/bin/orca $tdir/$job.inp > $job.out 

# ORCA has finished here. Now copy important stuff back (xyz files, GBW files etc.). Add more here if needed.
echo "Contents of the tmp dir:"
ls $tdir

cp $tdir/$job.gbw . 
cp $tdir/$job.engrad . 
cp $tdir/$job_trj.xyz .
cp $tdir/$job.xyz .
cp $tdir/$job.opt .
cp $tdir/$job.densities .
