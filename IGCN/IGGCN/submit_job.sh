#!/bin/bash
#SBATCH --job-name=your_job             # Change as needed
#SBATCH --time=12:00:00
#SBATCH --account=ee-452
#SBATCH --qos=ee-452
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # Adjust CPU allocation if needed
#SBATCH --output=interactive_job.out    # Output log file
#SBATCH --error=interactive_job.err     # Error log file


source activate network


python3 main_igcn_simple.py
