#subimssion syntax: ./submitxtb_pure.sh "best_job" "xeon8" "outputdir2" 8 "script_file"


RUN_NUMBER=$1
PAR=$2
OUT=$3
SCRIPT=$5

# get the filename without the extension
JOB=${RUN_NUMBER}
PARTITION=${PAR}
TIME=2-00:00:00


#mkdir $OUT

PWD=`pwd`

SUBMIT=qsub.tmp

NCPUS=$4
MEM=4G

mkdir -p $OUT
cd $OUT

cat > $SUBMIT <<!EOF
#!/bin/sh
#SBATCH --job-name=$JOB
#SBATCH --cpus-per-task=$NCPUS
#SBATCH --ntasks=1
#SBATCH --error=$PWD/$JOB\_%j.err
#SBATCH --output=$PWD/$JOB\_%j.out
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --no-requeue
#SBATCH --mem=$MEM

ulimit -s unlimited

module use /home/energy/stly/modules/modules/all
module load xtb/6.2.3
source activate /home/energy/magstr/miniconda3/envs/GA

python $SCRIPT

!EOF

sbatch $SUBMIT

