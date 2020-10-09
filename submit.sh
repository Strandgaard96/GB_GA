#!bin/bash

smiles_file=$(basename $1)

SUBMIT=qsub.tmp

PWD=`pwd`
JOB_NAME=$(basename $PWD)

echo $JOB_NAME

PARTITION=shortcoms
TIME=24:00:00
NCPUS=12
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

cp $PWD/$smiles_file .

/home/julius/soft/miniconda3/envs/default/bin/python /home/julius/soft/GB-GA/GA_catalyst.py $1

tar -cvzf out.tar.gz comp* *.result
cp out.tar.gz $PWD

rm -rf /scratch/\$SLURM_JOB_ID

!EOF

sbatch $SUBMIT
