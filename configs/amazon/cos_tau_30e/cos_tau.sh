#!/bin/bash
#SBATCH --job-name=amazon_cos_tau_30e
#SBATCH --account=proj_1876
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/amazon/cos_tau_30e/cos_tau.yaml
