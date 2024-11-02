#!/bin/bash
#SBATCH --job-name=ert_hilbert
#SBATCH --output=%x_%j.out 
#SBATCH --time=24:00:00               
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8            
#SBATCH --mem=64G                     
#SBATCH --gres=gpu:1               

source ../.venv/bin/activate

python ./ERT_hilbert.py

# squeue -j 36561129

