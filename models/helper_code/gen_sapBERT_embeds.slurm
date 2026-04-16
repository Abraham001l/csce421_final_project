#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=gen_sapBERT       
#SBATCH --time=00:30:00              
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=2GB                    
#SBATCH --output=Example4Out.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue



