#!/bin/bash
#SBATCH --partition=Odyssey                 # Partition name
#SBATCH --gres=gpu:1                        # GPU request
#SBATCH --mem=300G                          # Memory request
#SBATCH --job-name=fdv-unet              # Job name
#SBATCH --output=/users/local/path/Test-GPU_%j.log # Standard output and error log (%j for jobid)
#SBATCH --error=logs/err_%A_%a.txt          # Error log
#SBATCH --exclude=sl-mee-br-207,sl-mee-br-208,sl-mee-br-209   # Exclure certains serveurs

# Variables pour éviter NVML init
export PYTORCH_NO_NVML=1
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# Activer l’environnement
export CONDARC=/Odyssey/private/p25denai/.condarc
export HOME=/Odyssey/private/p25denai
unset SLURM_MEM_PER_CPU

source /Odyssey/private/p25denai/miniforge3/etc/profile.d/conda.sh
conda activate fdv

cd /Odyssey/private/p25denai/4dvarnet-global-mapping

srun python main.py xp=glo12-sla-4th-fdv-unetSolver-oseWosse

