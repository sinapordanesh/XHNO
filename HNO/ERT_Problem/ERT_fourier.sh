#!/bin/bash
#SBATCH --job-name=r_fno_ert_pq
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

python ERT_fourier.py --fn fno_pq --train data/s3/trainingPQ.mat --test data/s3/testPQ.mat --eval data/s3/evalPQ.mat --ts 66000

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

# squeue -j 36882863

