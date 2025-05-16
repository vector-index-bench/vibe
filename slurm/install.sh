#!/bin/bash
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=nvme:32

srun ./install.sh --build-dir $LOCAL_SCRATCH
