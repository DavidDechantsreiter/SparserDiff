#!/bin/bash
#SBATCH --job-name=sparsediff_ego
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/hpham/SparseDiff/logs/%j.out

source /scratch/hpham/miniforge3/etc/profile.d/conda.sh
conda activate sparse

export LD_PRELOAD=/scratch/hpham/miniforge3/envs/sparse/lib/libgomp.so

cd /home/hpham/wpi-graph-ai-mqp-25-26/SparserDiff/sparse_diffusion

python3 main.py +experiment=ego dataset=ego train.n_epochs=100 +trainer.accelerator=gpu +trainer.devices=1 general.wandb=disabled general.name=ego_baseline general.use_novel_sampling=false
