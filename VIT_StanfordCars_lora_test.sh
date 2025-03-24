#!/bin/bash

#SBATCH --job-name=vit_test
#SBATCH --output=./res/vit_stanfordcars_lora_test_res.txt
#SBATCH --error=./res/vit_stanfordcars_lora_test_error.txt
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../miniconda3/etc/profile.d/conda.sh
conda activate py310

export PYTHONPATH=$(pwd)

python VIT_StanfordCars_lora_test.py
