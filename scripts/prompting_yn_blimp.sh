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

rm -rf ~/.cache/huggingface/datasets/generator/

# Instruct models

TEMPLATE="prompts/blimp/template_blimp_yn-chat_${TEMPLATE_NUM}.json"

PYTHONPATH=. python src/prompting_yn.py \
  -d blimp -t $TEMPLATE \
  -m meta-llama/Meta-Llama-3-70B-Instruct -q "4bit" \
  -b 32

PYTHONPATH=. python src/prompting_yn.py \
  -d blimp -t $TEMPLATE \
  -m mistralai/Mixtral-8x7B-Instruct-v0.1 -q "4bit" \
  -b 64

PYTHONPATH=. python src/prompting_yn.py \
  -d blimp -t $TEMPLATE \
  -m Qwen/Qwen2-57B-A14B-Instruct -q "4bit" \
  -b 64
  # -m Qwen/Qwen1.5-32B-Chat -q "4bit" \

# Base models

TEMPLATE="prompts/blimp/template_blimp_yn_${TEMPLATE_NUM}.json"

PYTHONPATH=. python src/prompting_yn.py \
  -d blimp -t $TEMPLATE \
  -m meta-llama/Meta-Llama-3-70B -q "4bit" \
  -b 32

PYTHONPATH=. python src/prompting_yn.py \
  -d blimp -t $TEMPLATE \
  -m mistralai/Mixtral-8x7B-v0.1 -q "4bit" \
  -b 64

PYTHONPATH=. python src/prompting_yn.py \
  -d blimp -t $TEMPLATE \
  -m Qwen/Qwen2-57B-A14B -q "4bit" \
  -b 64
  # -m Qwen/Qwen1.5-32B -q "4bit" \
