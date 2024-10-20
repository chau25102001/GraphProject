#!/bin/bash
# SBATCH --job-name=nmduong        # Job name
#SBATCH --output=results_embeddings_h.txt      # Output file
#SBATCH --error=error_embeddings_h.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                              # Number of CPU cores per task                   │····································································································································

python train.py --config="configs/chet_use_text_embeddings_h.yaml"