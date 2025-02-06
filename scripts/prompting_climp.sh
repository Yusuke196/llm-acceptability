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
#   -d climp -t "prompts/climp/template_climp_ab-chat_${TEMPLATE_NUM}.json" \
#   -m 01-ai/Yi-1.5-34B-Chat -q "4bit" \
#   --max_new_tokens 1 -b 64  --use_outlines --debug
#   # -m Qwen/Qwen2-0.5B-Instruct -q "4bit" \

# exit 0

# Base models

PYTHONPATH=. python src/inference.py \
  -d climp -t "prompts/climp/template_climp_ab_${TEMPLATE_NUM}.json" \
  -m meta-llama/Meta-Llama-3-70B -q "4bit" \
  --max_new_tokens 1 -b 32 --use_outlines

PYTHONPATH=. python src/inference.py \
  -d climp -t "prompts/climp/template_climp_ab_${TEMPLATE_NUM}.json" \
  -m 01-ai/Yi-1.5-34B -q "4bit" \
  --max_new_tokens 1 -b 64 --use_outlines

PYTHONPATH=. python src/inference.py \
  -d climp -t "prompts/climp/template_climp_ab_${TEMPLATE_NUM}.json" \
  -m Qwen/Qwen2-57B-A14B -q "4bit" \
  --max_new_tokens 1 -b 64 --use_outlines

# Instruct models

PYTHONPATH=. python src/inference.py \
  -d climp -t "prompts/climp/template_climp_ab-chat_${TEMPLATE_NUM}.json" \
  -m meta-llama/Meta-Llama-3-70B-Instruct -q "4bit" \
  --max_new_tokens 1 -b 32 --use_outlines

PYTHONPATH=. python src/inference.py \
  -d climp -t "prompts/climp/template_climp_ab-chat_${TEMPLATE_NUM}.json" \
  -m 01-ai/Yi-1.5-34B-Chat -q "4bit" \
  --max_new_tokens 1 -b 64 --use_outlines

PYTHONPATH=. python src/inference.py \
  -d climp -t "prompts/climp/template_climp_ab-chat_${TEMPLATE_NUM}.json" \
  -m Qwen/Qwen2-57B-A14B-Instruct -q "4bit" \
  --max_new_tokens 1 -b 64 --use_outlines
