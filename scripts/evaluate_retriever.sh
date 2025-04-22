#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=EvalRetriever
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm/evaluate_retriever_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate bret

cd $HOME/bayesian-dpr

python src/eval_retriever.py