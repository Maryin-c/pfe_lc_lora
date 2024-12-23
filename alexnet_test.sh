#!/bin/bash
#SBATCH --job-name=alexnettest
#SBATCH --output=job_output_vit.txt
#SBATCH --error=job_error_vit.txt
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

srun source /miniconda3/etc/profile.d/conda.sh

srun conda activate py310

srun alexnet-test_with_old_lc.py
