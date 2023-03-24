#!/bin/bash
#SBATCH -Jtrain_CNN
#SBATCH --account=gts-jw254-coda20
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH -t23:59:59
#SBATCH -qinferno
#SBATCH -oReport-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiabin.phy@gmail.com

#cd $SLURM_SUBMIT_DIR
module load anaconda3
module load tensorflow-gpu/2.9.0
module load keras/2.9.0

conda activate SETI
srun python CNN.py
