#!/bin/bash
#SBATCH --job-name=llm_la
#SBATCH --partition=gpu_long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate llm-acceptability

TEMPLATE_NUM=$1
if [[ -z $1 ]]; then
  echo >&2 "Supply TEMPLATE_NUM"
  exit 1
fi

# rm -rf ~/.cache/huggingface/datasets/generator/

# Debug

# PYTHONPATH=. python src/inference.py \
#   -d blimp -t "prompts/blimp/template_blimp_ab-chat_${TEMPLATE_NUM}.json" \
#   -m Qwen/Qwen2-0.5B-Instruct -q "4bit" \
#   --max_new_tokens 1 -b 64 --use_outlines --debug
# #   -m meta-llama/Meta-Llama-3-8B-Instruct -q "4bit" \

# exit 0

# Base models

TEMPLATE="prompts/blimp/template_blimp_ab_${TEMPLATE_NUM}.json"

PYTHONPATH=. python src/inference.py \
  -d blimp -t $TEMPLATE \
  -m meta-llama/Meta-Llama-3-70B -q "4bit" \
  --max_new_tokens 1 -b 32 --use_outlines

PYTHONPATH=. python src/inference.py \
  -d blimp -t $TEMPLATE \
  -m mistralai/Mixtral-8x7B-v0.1 -q "4bit" \
  --max_new_tokens 1 -b 64 --use_outlines

PYTHONPATH=. python src/inference.py \
  -d blimp -t $TEMPLATE \
  -m Qwen/Qwen2-57B-A14B -q "4bit" \
  --max_new_tokens 1 -b 64 --use_outlines
  # -m Qwen/Qwen1.5-32B -q "4bit" \

# Instruct models

TEMPLATE="prompts/blimp/template_blimp_ab-chat_${TEMPLATE_NUM}.json"

PYTHONPATH=. python src/inference.py \
  -d blimp -t $TEMPLATE \
  -m meta-llama/Meta-Llama-3-70B-Instruct -q "4bit" \
  --max_new_tokens 1 -b 32 --use_outlines

PYTHONPATH=. python src/inference.py \
  -d blimp -t $TEMPLATE \
  -m mistralai/Mixtral-8x7B-Instruct-v0.1 -q "4bit" \
  --max_new_tokens 1 -b 64 --use_outlines

PYTHONPATH=. python src/inference.py \
  -d blimp -t $TEMPLATE \
  -m Qwen/Qwen2-57B-A14B-Instruct -q "4bit" \
  --max_new_tokens 1 -b 64 --use_outlines
  # -m Qwen/Qwen1.5-32B-Chat -q "4bit" \
