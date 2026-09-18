#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=glorys6h-v2
#SBATCH --output=/Odyssey/private/p23leflo/projects/4dvarnet-global-mapping/data/logs/linear6h_%A_%a.log
set -euo pipefail
export PYTHONNOUSERSITE=1
export XDG_CONFIG_HOME="$(mktemp -d /tmp/glorys6h-config.XXXXXX)"
export MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
cd /Odyssey/private/p23leflo/projects/4dvarnet-global-mapping/data
py=/Odyssey/private/p23leflo/miniforge3/envs/4dvarnet/bin/python
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  srun "$py" -u glorys6h_linear_v2.py --year "$((2010+SLURM_ARRAY_TASK_ID))"
else
  srun "$py" -u glorys6h_linear_v2.py "$@"
fi
