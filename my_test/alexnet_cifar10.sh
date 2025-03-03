#!/bin/bash

#SBATCH --job-name=alexnet_cifar
#SBATCH --output=./my_test/res/alexnet_cifar10_res_append.txt
#SBATCH --error=./my_test/res/alexnet_cifar10_error_append.txt
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../miniconda3/etc/profile.d/conda.sh
conda activate py310

export PYTHONPATH=$(pwd)

python ./my_test/alexnet_cifar10.py
