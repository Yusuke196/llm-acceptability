#!/bin/bash
#SBATCH --job-name=llm_la
#SBATCH --partition=gpu_long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate llm-acceptability

# rm -rf ~/.cache/huggingface/datasets/generator/

# Base models

PYTHONPATH=. python src/pred_by_prob.py -d climp \
  -a ll -m Qwen/Qwen2-57B-A14B -q "4bit" \
  -b 64
PYTHONPATH=. python src/pred_by_prob.py -d climp \
  -a ll -m 01-ai/Yi-1.5-34B -q "4bit" \
  -b 64
PYTHONPATH=. python src/pred_by_prob.py -d climp \
  -a ll -m meta-llama/Meta-Llama-3-70B -q "4bit" \
  -b 64

# Instruct models

PYTHONPATH=. python src/pred_by_prob.py -d climp \
  -a ll -m Qwen/Qwen2-57B-A14B-Instruct -q "4bit" \
  -b 64
PYTHONPATH=. python src/pred_by_prob.py -d climp \
  -a ll -m 01-ai/Yi-1.5-34B-Chat -q "4bit" \
  -b 64
PYTHONPATH=. python src/pred_by_prob.py -d climp \
  -a ll -m meta-llama/Meta-Llama-3-70B-Instruct -q "4bit" \
  -b 64
