#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=magstr@dtu.dk  # The default value is the submitting user.
#SBATCH --partition=xeon40
#SBATCH -N 1      # Minimum of 2 nodes
#SBATCH -n 40     # 24 MPI processes per node, 48 tasks in total, appropriate for xeon24 nodes
#SBATCH --time=2-02:00:00
#SBATCH --mem=350G

# Loading necessary modules
module load ORCA
module load OpenMPI

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
cp $tdir/$job.loc .
cp $tdir/$job.qro . 
cp $tdir/$job.uno .
cp $tdir/$job.unso .
cp $tdir/$job.opt .
cp $tdir/$job.densities .
cp $tdir/$job.uco . 
cp $tdir/$job.hess .
cp $tdir/$job.cis . 
cp $tdir/$job.dat . 
cp $tdir/$job.mp2nat . 
cp $tdir/$job_property.txt $SLURM_SUBMIT_DIR
cp $tdir/*spin* $SLURM_SUBMIT_DIR