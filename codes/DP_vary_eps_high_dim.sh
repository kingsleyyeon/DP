#!/bin/bash
#SBATCH --job-name=dp_vary_eps_highdim
#SBATCH --output=slurm_out_%A_%a.out
#SBATCH --error=slurm_err_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --array=0-6
#SBATCH --partition=caslake
#SBATCH --account=pi-sagnik

# Load Conda
source /home/sagnik/miniconda3/etc/profile.d/conda.sh
conda activate DPTransformer

# Navigate to project directory
cd /home/sagnik/Research/DP_transformer

# Define list of N values
N_list=(1000 1500 2000 2500 3000 3500 4000)

# Get N based on SLURM task ID
N=${N_list[$SLURM_ARRAY_TASK_ID]}

# Run the script
echo "=== Starting N=${N} on node $(hostname) ==="
python DP_eps_high_dim.py $N
