#!/bin/bash
#SBATCH --job-name=ERT_fourier
#SBATCH --output=%x_%j.out 
#SBATCH --time=24:00:00               
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8            
#SBATCH --mem=64G                     
#SBATCH --gres=gpu:1  

# ---------------------------------------------------------------------
echo "Starting run at: `date`"
# ---------------------------------------------------------------------

source ../.venv/bin/activate
python ./ERT_fourier.py

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

# squeue -j 36574394

