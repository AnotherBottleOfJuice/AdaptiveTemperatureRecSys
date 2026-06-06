#!/bin/bash
#SBATCH --job-name=test_exp                # Название задачи
#SBATCH --account=proj_1876                    # Идентификатор проекта
#SBATCH --time=0:10:00                         # Максимальное время выполнения (1 час)
#SBATCH --nodes=1                              # Требуемое кол-во узлов
#SBATCH --cpus-per-task=1                      # Количество CPU на одну задачу
#SBATCH --gpus=1                               # Требуемое кол-во GPU
#SBATCH --mail-type=BEGIN,END,FAIL             # Уведомления при старте, завершении или сбое задачи
#SBATCH --mail-user=maanbessolitsyn@edu.hse.ru # Ваша почта
#SBATCH --partition=test

module load Python/Anaconda_v11.2021  # Загрузка модуля Anaconda
source activate adaptivetemperaturerecsys
python main.py --config configs/test.yaml
