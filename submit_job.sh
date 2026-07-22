#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --job-name=training_multi_time_series
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

# Your commands go here
source ~/.bashrc
conda activate causal_rivers
python training.py --nodelist=gaia3

