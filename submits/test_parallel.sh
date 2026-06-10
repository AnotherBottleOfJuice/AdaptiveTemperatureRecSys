#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=proj_1876
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru

module load Python
source activate adaptivetemperaturerecsys
python main.py configs/test.yaml
