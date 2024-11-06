#!/bin/bash
#SBATCH --job-name=LNO_training
#SBATCH --output=%x_%j.out 
#SBATCH --time=8:00:00               # Time limit: 8 hours
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                     
#SBATCH --gres=gpu:1               

source ../.venv/bin/activate

python ./main.py

# squeue -j 36203770

