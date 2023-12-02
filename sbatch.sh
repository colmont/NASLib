#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --output=log.out

source /cluster/home/cdoumont/miniconda3/etc/profile.d/conda.sh
conda activate naslib
cd /cluster/home/cdoumont/NASLib
bash playground/scripts/run_bo.sh