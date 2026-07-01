#!/bin/bash
#SBATCH --job-name=shifted_cos_per_user_tau_30e_part3
#SBATCH --account=proj_1876
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/yandex/shifted_cos_per_user_tau_30e_part3/shifted_cos_per_user_tau.yaml
