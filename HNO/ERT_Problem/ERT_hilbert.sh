#!/bin/bash
#SBATCH --job-name=r_hno0_ert_k
#SBATCH --output=%x_%j.out 
#SBATCH --time=4:00:00               
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8            
#SBATCH --mem=64G                     
#SBATCH --gres=gpu:1               

# ---------------------------------------------------------------------
echo "Starting run at: `date`"
# ---------------------------------------------------------------------

source $HOME/repos/XNO/.venv/bin/activate

python ERT_hilbert.py --fn hno0_k --train data/s1/trainingK.mat --test data/s1/testK.mat --eval data/s1/evalK.mat --ts 80000

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------


# squeue -j 37231976

