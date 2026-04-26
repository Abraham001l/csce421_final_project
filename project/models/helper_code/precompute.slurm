#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=precompute
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # <--- This requests 8 cores for your 1 $
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=precompute.%j

cd ..
cd ..
cd ..
# added -nv flag to use gpus
# running exec rather then shell since no user input in compute node
singularity exec --nv \
  --bind "${PWD}/project:/project" \
  --pwd /project/models/helper_code \
  csce421_final_project.sif \
  python -u precompute.py
