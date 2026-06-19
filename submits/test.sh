#!/bin/bash
#SBATCH --job-name=test
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
python main.py configs/test.yaml
