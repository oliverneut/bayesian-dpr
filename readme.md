# Bayesian Dense Passage Retrieval (DPR)

This repository implements a Bayesian approach to Dense Passage Retrieval (DPR) with knowledge distillation capabilities. The implementation focuses on training a student model that learns from a teacher model while incorporating Bayesian uncertainty estimates.

## Overview

The project implements two main training approaches:
1. Standard DPR training
2. Knowledge distillation with Bayesian uncertainty

The implementation uses PyTorch and includes features like:
- Learning rate scheduling with warmup
- Model checkpointing
- Validation metrics tracking (nDCG and MRR)
- Integration with Weights & Biases for experiment tracking

## Installation

Create and activate the conda environment:
```bash
conda env create -f env.yml
conda activate bayesian-dpr
```

## Download MS-MARCO

Create a `data` folder under the root folder and download MS-MARCO. We also use the hard negatives from the SentenceTransformers repository.
```bash
mkdir data
cd data
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
unzip msmarco.zip
wget https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz
```

To prepare the data for training, run:
```bash
python prepare_data.py
```

## Configuration

All model parameters and training settings can be configured in `config.yml`. Key parameters include:

```yaml
train:
  knowledge_distillation: True  # Enable/disable knowledge distillation
  model_name: bert-base         # Student model architecture
  teacher_model_name: bert-base-msmarco  # Teacher model architecture
  max_qry_len: 32              # Maximum query length
  max_psg_len: 256            # Maximum passage length
  alpha: 0.0                  # Knowledge distillation weight
  prior_scale: 1.0            # Prior scale for Bayesian model
  wishart_scale: 0.001        # Wishart scale for Bayesian model
  batch_size: 16              # Training batch size
  num_epochs: 4               # Number of training epochs
  lr: 5.0e-6                  # Initial learning rate
  min_lr: 5.0e-8              # Minimum learning rate
  warmup_rate: 0.1            # Learning rate warmup proportion
```

## Usage

The main training script is `src/knowledge_distillation.py`. To run training:

```bash
python src/knowledge_distillation.py
```

The script will:
1. Load the configuration from `src/utils/config.yml`
2. Initialize the appropriate trainer (DPR or Knowledge Distillation)
3. Train the model with the specified parameters
4. Save checkpoints and log metrics to Weights & Biases

## Model Architecture

The implementation supports two types of models:
1. Standard DPR models
2. Variational Bayesian models with uncertainty estimates

The Bayesian models use a diagonal parameterization by default and incorporate prior and Wishart scales for uncertainty estimation.

## Model Naming

Model checkpoints are saved with names that reflect their configuration parameters. The naming convention follows this pattern:
```
{ckpt_filename}-{alpha}-{prior_scale}-{wishart_scale}.pt
```

For example, with the default configuration:
- `ckpt_filename`: "vbll-kd"
- `alpha`: 0.0
- `prior_scale`: 1.0
- `wishart_scale`: 0.001

The resulting model name would be: `vbll-kd-0.0-1.0-0.001.pt`

These parameters can be adjusted in `src/utils/config.yml` to create different model variants.

## Evaluation

The model is evaluated using:
- nDCG@k (Normalized Discounted Cumulative Gain)
- MRR@k (Mean Reciprocal Rank)

These metrics are computed on a validation set and logged during training.

## Scripts

The `scripts` directory contains shell scripts for running different components of the project on a SLURM cluster:

- `knowledge_distillation.sh`: Runs the knowledge distillation training process
  - Uses 1 H100 GPU
  - Allocates 18 CPUs
  - Time limit: 7 hours
  - Output logs saved to `slurm/knowledge_distillation/`

- `eval_retriever.sh`: Runs the evaluation of the trained model
  - Uses 1 H100 GPU
  - Allocates 18 CPUs
  - Time limit: 4 hours
  - Output logs saved to `slurm/eval_retriever/`

- `prepare_data.sh`: Script for data preparation
- `download_data.sh`: Script for downloading the MS-MARCO dataset
- `install_env.sh`: Script for setting up the conda environment

These scripts are configured for the Snellius cluster environment and include appropriate SLURM directives for resource allocation.

## License

[Add your license information here]
