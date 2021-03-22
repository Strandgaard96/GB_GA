#!bin/bash

SUBMIT=qsub.tmp

PWD=`pwd`
JOB_NAME='GBGA_0432'
outfile=log_GBGA.tar.gz

PARTITION=shortcoms
TIME=168:00:00
NCPUS=$1
MEM=12GB

cat > $SUBMIT <<!EOF
#!/bin/sh
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$NCPUS
#SBATCH --mem=$MEM
#SBATCH --ntasks=1
#SBATCH --error=$PWD/$JOB_NAME\_%j.err
#SBATCH --output=$PWD/$JOB_NAME\_%j.out
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --no-requeue

# Create scratch folder
mkdir /scratch/\$SLURM_JOB_ID
cd /scratch/\$SLURM_JOB_ID

cp /home/julius/soft/GB-GA/ZINC_1000_amines.smi .
cp /home/julius/soft/GB-GA/neutralize.json .

# xtb
export XTBHOME=/home/julius/soft/xtb-6.3.3/bin
ulimit -s unlimited

/home/julius/soft/miniconda3/envs/default/bin/python /home/julius/soft/GB-GA/GB-GA_catalyst/multiple_GA_catalyst.py $PWD $1 /scratch/\$SLURM_JOB_ID/all_generations/

tar -czf $outfile -C /scratch/\$SLURM_JOB_ID/ .
cp $outfile $PWD

cd ..

# Remove scratch folder
rm -rf /scratch/\$SLURM_JOB_ID

!EOF

sbatch $SUBMIT
