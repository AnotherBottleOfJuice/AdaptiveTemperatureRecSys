#!/bin/bash
#SBATCH --job-name=amazon_linear_tau_30e_part2
#SBATCH --account=proj_1876
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/amazon/linear_tau_30e_part2/linear_tau_30e_part2.yaml
