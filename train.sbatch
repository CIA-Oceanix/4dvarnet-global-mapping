#!/bin/bash
#SBATCH --partition=Odyssey
#SBATCH --nodelist=sl-mee-br-215
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=downsampling
#SBATCH --output=gpu_test_%j.log
#SBATCH --error=gpu_test_%j.err
#SBATCH --error=logs/err_%A_%a.txt          # Error log




export HOME=/Odyssey/private/$USER
source $HOME/.bashrc





# Variables pour éviter NVML init
# export PYTORCH_NO_NVML=1
# export NVIDIA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0

# Activer l’environnement
# export CONDARC=/Odyssey/private/p25denai/.condarc
# export HOME=/Odyssey/private/p23leflo
# unset SLURM_MEM_PER_CPU

# source /Odyssey/private/p23leflo/miniforge3/etc/profile.d/conda.sh
conda activate 4dvarnet

cd /Odyssey/private/p23leflo/projects/4dvarnet-global-mapping

HYDRA_FULL_ERROR=1 python main.py xp=SLA-MAPPING/unet_unrolling_5iter_grad_downsampling ckpt=outputs/2026-05-04/glo12-sla-4th-unet15M-unrolling-aeunet_64_2_96_3-randinit-5iter-grad-ds4-ossev2-noiseg+u03-17-36-58/SLA-MAPPING/unet_unrolling_5iter_grad_downsampling/checkpoints/last.ckpt