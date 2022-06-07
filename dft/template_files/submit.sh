#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --partition=xeon24
#SBATCH -N 1  # 
#SBATCH -n 24 #
#SBATCH --time=2-00:00:00
#SBATCH --mem=250G     # RAM per node: 23G on xeon8, 63G on xeon16, 250G on xeon24, 350G on xeon40 

module load AMS/2020
export SCM_TMPDIR=/scratch/magstr/
export SCM_IOBUFFERSIZE=4096 # size in MB of a buffer in memory to avoid some I/O to disk.  Recommended values: 512 on xeon8, 1024 on xxeon16, 2048 on xeon24, 4096 on xeon40.

folder=$1 # Folder to go for calculations. the xyz file should already be here
job=$2 #Path to ams run file, without the extension

# Preprocessing python script to handle input files
python_script=../pre_processing/constraints.py

chmod u+x $job.run

rm -r $folder/ams.results # remove previous results, otherwise will not run !
cp $job.run $folder; cp $python_script $folder; cd $folder

# Preprocessing
python constraints.py $job.run

./$job.run > $job.out

cd ..
rm -rf /scratch/magstr/* # clean scratch space on compute node

