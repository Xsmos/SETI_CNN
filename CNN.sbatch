#!/bin/bash
#SBATCH --job-name="CNN"
#SBATCH --output=CNN.out
#SBATCH -phive-gpu
#SBATCH -N1
#SBATCH --mem-per-gpu=128G
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -A hive-jw254
#SBATCH -t 48:00:00

module load anaconda3
module load tensorflow-gpu/2.9.0
module load keras/2.9.0

conda activate SETI
srun python CNN.py
