#!/bin/bash
#SBATCH --job-name=hilbert_2d_ert_s1
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

python ./ERT_hilbert.py

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------


# squeue -j 36871631, 36871852

