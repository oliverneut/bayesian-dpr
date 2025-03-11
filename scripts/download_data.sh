#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=DownloadData
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=slurm/download_data_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

cd $HOME/bayesian-dpr

mkdir data
cd data
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
unzip msmarco.zip
wget https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz