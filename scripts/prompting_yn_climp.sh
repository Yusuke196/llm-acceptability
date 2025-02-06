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

# PYTHONPATH=. python src/prompting_yn.py \
#   -d climp -t "prompts/climp/template_climp_yn-en-chat_${TEMPLATE_NUM}.json" \
#   -m Qwen/Qwen2-0.5B-Instruct -q "4bit" \
#   -b 64 --debug
  # -d climp -t "prompts/climp/template_climp_yn-chat_${TEMPLATE_NUM}.json" \
  # -m 01-ai/Yi-1.5-34B -q "4bit" \
  # -m meta-llama/Meta-Llama-3-8B -q "4bit" \

# exit 0

# Base models

# TEMPLATE_TYPE=yn
TEMPLATE_TYPE=yn-en

BASE_MODEL=meta-llama/Meta-Llama-3-70B
# BASE_MODEL=Qwen/Qwen2-57B-A14B
# BASE_MODEL=01-ai/Yi-1.5-34B

PYTHONPATH=. python src/prompting_yn.py \
  -d climp -t "prompts/climp/template_climp_${TEMPLATE_TYPE}_${TEMPLATE_NUM}.json" \
  -m $BASE_MODEL -q "4bit" \
  -b 64

# Instruct models

# INSTRUCT_MODEL=meta-llama/Meta-Llama-3-70B-Instruct # Requires batch size 32
# INSTRUCT_MODEL=Qwen/Qwen2-57B-A14B-Instruct
# INSTRUCT_MODEL=01-ai/Yi-1.5-34B-Chat

# PYTHONPATH=. python src/prompting_yn.py \
#   -d climp -t "prompts/climp/template_climp_${TEMPLATE_TYPE}-chat_${TEMPLATE_NUM}.json" \
#   -m $INSTRUCT_MODEL -q "4bit" \
#   -b 64
