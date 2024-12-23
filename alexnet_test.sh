#!/bin/bash

#SBATCH --job-name=alexnettest
#SBATCH --output=job_output_vit.txt
#SBATCH --error=job_error_vit.txt
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../miniconda3/etc/profile.d/conda.sh
conda activate py310

srun python alexnet-test_with_old_lc.py
srun sleep 20
