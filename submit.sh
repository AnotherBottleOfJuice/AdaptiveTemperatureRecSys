#!/bin/bash
#SBATCH --job-name=linear_tau_30e
#SBATCH --account=proj_1876
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru
#SBATCH --partition=test

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/linear_tau_30e.yaml
