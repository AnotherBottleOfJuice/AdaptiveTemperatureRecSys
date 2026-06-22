#!/bin/bash
#SBATCH --job-name=amazon_neg_u30k_ib10k
#SBATCH --account=proj_1876
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/amazon/negatives_sweep/neg_u30k_ib10k.yaml
