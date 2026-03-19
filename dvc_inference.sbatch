#!/bin/bash

#SBATCH --partition=Odyssey                       
#SBATCH --gres=gpu:h200:1                       # a100:1                       
#SBATCH --job-name=inference                   
#SBATCH --output=logs/Test-GPU_%j.log           
#SBATCH --mem=100G
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --cpus-per-gpu=16

# Activer l’environnement
export CONDARC=/Odyssey/private/rfablet/miniforge3/.condarc
export HOME=/Odyssey/private/rfablet
# unset SLURM_MEM_PER_CPU

#source $HOME/.bashrc
source "/Odyssey/private/rfablet/miniforge3/etc/profile.d/conda.sh"
conda activate fdv

export scrdir=/SCRATCH/rfablet/Dataset/
if [ ! -d $scrdir ]; then
    mkdir -p $scrdir;
fi;


cd /Odyssey/private/rfablet/Python/4dvarnet-global-mapping/inference
#srun python main.py xp=glo12-sla-4th-unet domain=glorys
#srun python main.py xp=glo12-sla-4th-unet-oseWosse domain=glorys
#srun python main.py xp=glo12-sla-4th-unet+-oseWosse domain=glorys
srun dvc repro
