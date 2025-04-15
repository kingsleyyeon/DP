#!/bin/bash
#SBATCH --job-name=dp_transformer
#SBATCH --output=slurm_out_%A_%a.out
#SBATCH --error=slurm_err_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --array=0-11
#SBATCH --partition=caslake
#SBATCH --account=pi-sagnik

# Load Conda (no module needed)
source /home/sagnik/miniconda3/etc/profile.d/conda.sh
conda activate DPTransformer

# Go to your project directory
cd /home/sagnik/Research/DP_transformer

echo "=== Starting job ID ${SLURM_ARRAY_TASK_ID} on node $(hostname) ==="

# Define parameter lists (6 p values × 2 c values = 12 combinations)
p_list=(2.0 2.02 2.04 2.06 2.08 2.1)
c_list=(2 4)

# Compute indices
idx=$SLURM_ARRAY_TASK_ID
p_index=$((idx / 2))
c_index=$((idx % 2))

p=${p_list[$p_index]}
c=${c_list[$c_index]}

# Pass p and c as arguments
python DP_Robustness.py $p $c