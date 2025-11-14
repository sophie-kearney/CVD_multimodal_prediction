#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=tabPFN_EHR
#SBATCH --error=logs/tabPFN_EHR_%j.err
#SBATCH --output=logs/tabPFN_EHR_%j.out

echo "> START"
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
echo "> ACTIVATING VENV"
conda activate llm_env
echo "> SET UP CUDA"
module unload cuda
module load cuda/11.8

echo "> RUNNING TABPFN"
python analysis/models/tabPFN.py

echo ">END"