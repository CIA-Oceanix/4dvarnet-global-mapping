#!/bin/bash
#SBATCH --partition=Odyssey                 # Partition name
#SBATCH --gres=gpu:h100:1                        # GPU request
#SBATCH --cpus-per-task=12 
#SBATCH --mem=96G                          # Memory request
#SBATCH --job-name=unet3_unrolling             # Job name
#SBATCH --output=/users/local/path/Test-GPU_%j.log # Standard output and error log (%j for jobid)
#SBATCH --error=logs/err_%A_%a.txt          # Error log





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

cd /Odyssey/private/p25denai/4dvarnet-ronan_dev/

#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_1iter_obs
#HYDRA_FULL_ERROR=1 python main.py xp=unet_obs
#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_5iter_gradsplit
#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_5iter_grad+state
HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_5iter_grad
#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_5iter_obs+state
