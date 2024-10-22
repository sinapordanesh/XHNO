#!/bin/bash
#SBATCH --job-name=LNO_training
#SBATCH --output=LNO_training_%j.out  # Save output to a file named LNO_training_JOBID.out
#SBATCH --error=LNO_training_%j.err   # Save error messages to a file named LNO_training_JOBID.err
#SBATCH --time=48:00:00               # Time limit: 48 hours
#SBATCH --ntasks=1                    # Number of tasks (usually 1 if you are not parallelizing with MPI)
#SBATCH --cpus-per-task=8             # Number of CPU cores per task (adjust based on your requirement)
#SBATCH --gpus=1                      # Request 1 GPU
#SBATCH --mem=32G                     # Memory per node (adjust based on your requirement)
#SBATCH --partition=gpu:1               # Request GPU partition (adjust based on available partitions)
#SBATCH --mail-type=BEGIN,END,FAIL     # Get an email when the job begins, ends, or fails
#SBATCH --mail-user=myhomeqrc@gmail.com  # Set your email

source ../.venv/bin/activate

python ./main.py

