#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:40:00
#SBATCH --output=slurm/install_environment/%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

# conda env remove --name bret
# Activate your environment
conda env create -f env.yml
# conda activate bret