#!/bin/bash

#SBATCH --partition=Odyssey                       
#SBATCH --gres=gpu:a100:1                       
#SBATCH --job-name=unet-4th                     
#SBATCH --output=logs/Test-GPU_%j.log           
#SBATCH --mem=200G
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --cpus-per-gpu=16

# Activer l’environnement
export CONDARC=/Odyssey/private/rfablet/miniforge3/.condarc
export HOME=/Odyssey/private/rfablet
# unset SLURM_MEM_PER_CPU

source $HOME/.bashrc
# source /Odyssey/private/rfablet/miniforge3/etc/profile.d/conda.sh
conda activate fdv

# scrdir=/SCRATCH/rfablet/Dataset/
# mkdir -p $scrdir
# cp --update=none /Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_4th.nc $scrdir/ #glorys4_2010_2019_sla.nc $scrdir/
# cp --update=none /Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_4th_input.nc $scrdir/ #glorys4_2010_2019_sla.nc $scrdir/
# cp --update=none -r /Odyssey/public/data_challenge_ssh_ose/data/alongtrack/mask_6nadirs_4th.nc $scrdir/

cd /Odyssey/private/rfablet/Python/4dvarnet-global-mapping
srun  python main.py main.py xp=glo12-sla-4th-unet domain=glorys

