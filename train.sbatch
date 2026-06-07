#!/bin/bash

#SBATCH --partition=Odyssey_GPU                   
#SBATCH --gres=gpu:h100:1                       # a100:1  
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --job-name=downsampling
#SBATCH --output=logs/Test-GPU_%j.log
#SBATCH --error=logs/err_%A_%a.txt




export HOME=/Odyssey/private/$USER
source $HOME/.bashrc
conda activate 4dvarnet

cd /Odyssey/private/p23leflo/projects/4dvarnet-global-mapping

HYDRA_FULL_ERROR=1 python main.py xp=SLA-MAPPING/unet_unrolling_5iter_grad_downsampling_insolver
