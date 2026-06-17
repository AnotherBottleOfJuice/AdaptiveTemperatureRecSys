#!/bin/bash
#SBATCH --job-name=cos_per_user_tau_30e
#SBATCH --account=proj_1876
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/cos_per_user_tau_30e.yaml
