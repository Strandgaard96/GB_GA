#!/bin/bash
#SBATCH --job-name=orca
#SBATCH --nodes=1
#SBATCH -e job-%j.er
#SBATCH -o job-%j.out
#SBATCH -n 40
#SBATCH --mem=250GB
#SBATCH -t 10000:00:00
#SBATCH --partition kemi1

# get the filename without the extension
JOB=orca
PWD=`pwd`

export PATH=/software/kemi/Orca/orca_5_0_1_linux_x86-64_openmpi411:/software/kemi/openmpi/openmpi-4.1.1/bin:$PATH
export LD_LIBRARY_PATH=/software/kemi/openmpi/openmpi-4.1.1/lib:$LD_LIBRARY_PATH
export ORCA=/software/kemi/Orca/orca_5_0_1_linux_x86-64_openmpi411/orca

SCRATCH=/scratch/$SLURM_JOB_ID
# make scratch dir
mkdir -p $SCRATCH || exit 0

echo "Running on " hostname > $SLURM_SUBMIT_DIR/$JOB.out
date >> $SLURM_SUBMIT_DIR/$JOB.out
ldd $ORCA
cd $SCRATCH
cp $SLURM_SUBMIT_DIR/$JOB.inp .
cp $SLURM_SUBMIT_DIR/*.xyz .

$ORCA $JOB.inp >> $SLURM_SUBMIT_DIR/$JOB.out

echo "Contents of the tmp dir:"
ls $SCRATCH

# done, now copy back wanted files.
cp $JOB.xyz $SLURM_SUBMIT_DIR
#cp $JOB.gbw $SLURM_SUBMIT_DIR
#cp $JOB.densities $SLURM_SUBMIT_DIR
cp ${JOB}_trj.xyz $SLURM_SUBMIT_DIR
cp $JOB.opt $SLURM_SUBMIT_DIR
