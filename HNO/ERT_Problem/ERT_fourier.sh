#!/bin/bash
#SBATCH --job-name=r_fourier_2d_ert_s2_2
#SBATCH --output=%x_%j.out 
#SBATCH --time=6:00:00               
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8            
#SBATCH --mem=64G                     
#SBATCH --gres=gpu:1  

# ---------------------------------------------------------------------
echo "Starting run at: `date`"
# ---------------------------------------------------------------------

source $HOME/repos/XNO/.venv/bin/activate
python ./ERT_fourier.py

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

# squeue -j 36882863

