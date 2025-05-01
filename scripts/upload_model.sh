#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=UploadModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm/upload_model/%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate bret

cd $HOME/bayesian-dpr

python src/upload_model.py