#!/bin/bash
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

srun python3 run.py --dataset imagenet-clip-512-normalized --count 100 --parallelism 15
