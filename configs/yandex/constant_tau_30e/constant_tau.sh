#!/bin/bash
#SBATCH --job-name=constant_tau_30e
#SBATCH --account=proj_1876
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/yandex/constant_tau_30e/constant_tau.yaml
