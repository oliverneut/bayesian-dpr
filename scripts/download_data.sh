#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=DownloadData
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=slurm/download_data/%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

cd $HOME/bayesian-dpr

# NQ
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
# HotpotQA
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip
# FiQA18
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip


# mkdir data
cd data
# wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
# wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip
unzip hotpotqa.zip
# wget https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz