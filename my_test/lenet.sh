#!/bin/bash

#SBATCH --job-name=lenet
#SBATCH --output=./res/lenet_mnist_res.txt
#SBATCH --error=./res/lenet_mnist_error.txt
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../../miniconda3/etc/profile.d/conda.sh
conda activate py310

python ./my_test/lenet.py