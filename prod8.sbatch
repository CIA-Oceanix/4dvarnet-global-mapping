#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --job-name=ose8th
#SBATCH --output=logs/ose8th_%j.log
export HOME=/Odyssey/private/p23leflo
export PYTHONNOUSERSITE=1
source $HOME/.bashrc
conda activate 4dvarnet
mkdir -p /Odyssey/private/p23leflo/data/ose_8th
cd /Odyssey/private/p23leflo/projects/4dvarnet-global-mapping
python -u prod8.py
