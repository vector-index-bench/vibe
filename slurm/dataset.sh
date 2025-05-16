#!/bin/bash
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,nvme:250

export VIBE_CACHE=$LOCAL_SCRATCH

srun ./create_dataset.sh --singularity-args "--bind $LOCAL_SCRATCH:$LOCAL_SCRATCH --nv" --dataset imagenet-clip-512-normalized
