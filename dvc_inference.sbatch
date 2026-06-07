#!/bin/bash

#SBATCH --partition=Odyssey_GPU                      
#SBATCH --gres=gpu:h100:1                       # a100:1                       
#SBATCH --job-name=inference                   
#SBATCH --output=logs/Test-GPU_%j.log           
#SBATCH --mem=100G
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --cpus-per-task=4

# Activer l’environnement
export CONDARC=/Odyssey/private/p23leflo/miniforge3/.condarc
export HOME=/Odyssey/private/p23leflo
# unset SLURM_MEM_PER_CPU

source $HOME/.bashrc
conda activate 4dvarnet




cd /Odyssey/private/p23leflo/projects/4dvarnet-global-mapping/inference
#srun python main.py xp=glo12-sla-4th-unet domain=glorys
#srun python main.py xp=glo12-sla-4th-unet-oseWosse domain=glorys
#srun python main.py xp=glo12-sla-4th-unet+-oseWosse domain=glorys
HYDRA_FULL_ERROR=1 srun dvc repro
