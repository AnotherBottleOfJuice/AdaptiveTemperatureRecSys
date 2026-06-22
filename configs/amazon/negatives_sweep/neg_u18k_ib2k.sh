#!/bin/bash
#SBATCH --job-name=amazon_neg_u18k_ib2k
#SBATCH --account=proj_1876
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/amazon/negatives_sweep/neg_u18k_ib2k.yaml
