#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=RunKD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=18:00:00
#SBATCH --output=slurm/knowledge_distillation_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate bret

cd $HOME/bayesian-dpr

python src/knowledge_distillation.py