#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=train_lstm
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --output=train_lstm.%j

cd ..
cd ..
cd ..
# added -nv flag to use gpus
# running exec rather then shell since no user input in compute node
singularity exec --nv \
  --bind "${PWD}/project:/project" \
  --pwd /project/models/transformer_lstm \
  csce421_final_project.sif \
  python -u train.py
