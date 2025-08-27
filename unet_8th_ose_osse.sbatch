#!/bin/bash

#SBATCH --partition=Odyssey                       
#SBATCH --gres=gpu:h100:1                       
#SBATCH --job-name=unet-8th                     
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


cp --update=none /Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_8th_float32.nc $scrdir/ #glorys4_2010_2019_sla.nc $scrdir/
cp --update=none /Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_8th_input.nc $scrdir/ #glorys4_2010_2019_sla.nc $scrdir/
cp --update=none /Odyssey/public/altimetry_traces/2010_2019/gridded_0.125deg/sla_l3_all_2010_2019_0.125deg_convl4.nc $scrdir/
#cp --update=none /Odyssey/public/duacs/2010_2019/duacs_2010_2019_0.125deg.nc $scrdir/
cp --update=none /Odyssey/public/duacs/2010_2019/duacs_2010_2019_0.125deg_float32.nc $scrdir/

cd /Odyssey/private/rfablet/Python/4dvarnet-global-mapping
#srun  python main.py main.py xp=glo12-sla-4th-unet domain=glorys
srun python main.py xp=glo12-sla-8th-unet-oseWosse domain=glorys