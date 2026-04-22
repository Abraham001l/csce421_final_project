#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=train_tff_v1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # <--- This requests 8 cores for your 1 $
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=train_tff_v1.%j

cd ..
cd ..
cd ..
# added -nv flag to use gpus
# running exec rather then shell since no user input in compute node
singularity exec --nv \
  --bind "${PWD}/project:/project" \
  --pwd /project/models/transformer_ff_v1 \
  csce421_final_project.sif \
  python -u train.py
