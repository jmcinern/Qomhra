#!/bin/bash
#SBATCH --job-name=CPT_JM
#SBATCH --output=./out/qwen_mini_cpt_%j.out
#SBATCH --error=./err/qwen_mini_cpt_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=k2-gpu-v100  
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com
module load python3/3.10.5/gcc-9.3.0 # availible python
module load libs/nvidia-cuda/12.4.0/bin # cuda
module load openmpi #multi process
source /mnt/scratch2/users/40460549/cpt-dail/myenv_new/bin/activate
pip install --no-cache-dir -r "requirements.txt" # dependencies
export WANDB_API_KEY="" # logging
export HF_KEY="" # pushing to hf after running
cd $SLURM_SUBMIT_DIR                     # ensure we’re in the project dir


# to run
accelerate launch \
--num_processes 2 \
--mixed_precision bf16 \
cpt.py
