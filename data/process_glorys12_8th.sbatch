#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=glorys-8th
#SBATCH --output=logs/glorys8th_%A_%a.log
#SBATCH --error=logs/glorys8th_err_%A_%a.txt

# /homes illisible depuis les noeuds CPU (lecon infra 02/07)
export HOME=/Odyssey/private/$USER
export PYTHONNOUSERSITE=1

cd /Odyssey/private/p23leflo/projects/4dvarnet-global-mapping/data

PY=/Odyssey/private/p23leflo/miniforge3/envs/4dvarnet/bin/python
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    YEAR=$((2010 + SLURM_ARRAY_TASK_ID))
    srun $PY -u process_glorys12_8th.py --year "$YEAR"
else
    srun $PY -u process_glorys12_8th.py "$@"
fi
