#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=results_embeddings.txt      # Output file
#SBATCH --error=error_embeddings.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                              # Number of CPU cores per task                   │····································································································································

python initialize_embeddings.py