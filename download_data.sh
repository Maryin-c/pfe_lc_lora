#!/bin/bash

#SBATCH --job-name=download
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../miniconda3/etc/profile.d/conda.sh
conda activate py310

export PYTHONPATH=$(pwd)

python download_data.py
