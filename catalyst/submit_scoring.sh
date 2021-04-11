#!/bin/bash
scoring_script=$1
IN=$2
nconfs=$3
randomseed=$4
timing_logger=$5
warning_logger=$6
directory=$7
cpus_per_molecule=$8

# get the filename without the extension
JOB=${IN%.*}

PARTITION=mko # sauer or coms or teach
TIME=1:00:00

SUBMIT=qsub.tmp

PWD=`pwd`

NCPUS=1
MEM=1GB


cat > $SUBMIT <<!EOF
#!/bin/sh
#SBATCH --job-name=$JOB
#SBATCH --cpus-per-task=$NCPUS
#SBATCH --ntasks=1
#SBATCH --error=$PWD/$JOB.err
#SBATCH --output=$PWD/$JOB.out
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --no-requeue
#SBATCH --mem=$MEM

# Create scratch folder
mkdir /scratch/\$SLURM_JOB_ID
cd $PWD

export GAUSS_SCRDIR=/scratch/\$SLURM_JOB_ID
export GAUSS_EXEDIR=/opt/gaussian/g16/legacy/g16

python3 $scoring_script $IN $nconfs $randomseed $timing_logger $warning_logger $directory $cpus_per_molecule

# Remove scratch folder
rm -rf /scratch/\$SLURM_JOB_ID

!EOF

sbatch $SUBMIT

# MEM=1gb
#SBATCH --mem=$MEM
