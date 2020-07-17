#!/bin/bash

#SBATCH --account=def-abhamza
#SBATCH --mem=64GB 

#SBATCH --gres=gpu:v100l:1 
# SBATCH --gres=gpu:1 

#SBATCH --ntasks-per-node=8  
#SBATCH --time=00-10:00 
#SBATCH --output=./logs/log.out

# Load CUDA
module load cuda cudnn 
nvidia-smi

# Activate env
source /home/hasib/projects/def-abhamza/hasib/envs/gpu/bin/activate

# Launch script
cd scripts
python train_seg.py
# python train_classifier.py



# Run this on terminal
# sbatch job.sh

# Server configs
# --------------
# DATASET_FOLDER = "/home/hasib/scratch/npy_data" # use this when on server
# git fetch --all
# git reset --hard origin
