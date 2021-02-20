#!bin/bash

inp_file=$(basename $1)

SUBMIT=qsub.tmp

PWD=`pwd`
JOB_NAME='GBGA_0432'

PARTITION=coms
TIME=24:00:00
NCPUS=4
MEM=4GB

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

cp $PWD/$inp_file .
cp $PWD/sa/neutralize.json .
# cp /home/julius/soft/GB-GA/catalyst/structures/ts7/rr_dummy.sdf .
# cp $PWD/$smiles_file .
# cp $PWD/SA_scores.txt .
# cp $PWD/sa/fpscores.pkl.gz .

# xtb
export XTBHOME=/home/julius/soft/xtb-6.3.3/bin #/opt/xtb/6.1/xtb-190527/bin
ulimit -s unlimited
export OMP_STACKSIZE=6G
export OMP_NUM_THREADS=$NCPUS,1
export MKL_NUM_THREADS=$NCPUS


/home/julius/soft/miniconda3/envs/default/bin/python /home/julius/soft/GB-GA/GA_catalyst.py $1

tar -cvzf out.tar.gz comp* *.result
cp out.tar.gz $PWD

rm -rf /scratch/\$SLURM_JOB_ID

!EOF

sbatch $SUBMIT
