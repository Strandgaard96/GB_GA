#!/bin/bash
# Submit a batch of jobs by supplying path to folder

for f in /home/energy/magstr/nitrogenase/schrock/cycle_restart/*
do
  sbatch --job-name f $1 $2 $f
done

