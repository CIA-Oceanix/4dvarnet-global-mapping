#!/bin/bash
#SBATCH --partition=Odyssey_GPU                 # Partition name
#SBATCH --gres=gpu:h100:1                        # GPU request
#SBATCH --cpus-per-task=12 
#SBATCH --mem=96G                          # Memory request
#SBATCH --job-name=graph_global_sla             # Job name
#SBATCH --output=logs/Test-GPU_%j.log # Standard output and error log (%j for jobid)
#SBATCH --error=logs/err_%A_%a.txt          # Error log






# Variables pour éviter NVML init
# export PYTORCH_NO_NVML=1
# export NVIDIA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0

# Activer l’environnement

export HOME=/Odyssey/private/m24ouesl

unset SLURM_MEM_PER_CPU

source $HOME/.bashrc
conda activate 4dvarnet-gm
cd /Odyssey/private/m24ouesl/4dvarnet-global-mapping

#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_1iter_obs
HYDRA_FULL_ERROR=1 python main.py xp=SLA-MAPPING/graph_graphcast_direct
#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_5iter_gradsplit
#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_5iter_grad+state
#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_5iter_grad
#HYDRA_FULL_ERROR=1 python main.py xp=unet_unrolling_5iter_obs+state