#!/bin/bash
#SBATCH --job-name=llm_la
#SBATCH --partition=gpu_long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:6000:1
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate llm-acceptability

export TOKENIZERS_PARALLELISM=false

APPROACH=$1
if [[ -z $1 ]]; then
  echo >&2 "Supply APPROACH"
  exit 1
fi

TEMPLATE=$2
if [[ -z $2 ]]; then
  echo >&2 "Supply TEMPLATE"
  exit 1
fi

MODEL=$3
if [[ -z $3 ]]; then
  echo >&2 "Supply MODEL"
  exit 1
fi

BSIZE=$4
if [[ -z $4 ]]; then
  echo >&2 "Supply BSIZE"
  exit 1
fi

PYTHONPATH=. python src/pred_by_prob.py -d blimp \
  -a $APPROACH -t $TEMPLATE \
  -m $MODEL -q "4bit" \
  -b $BSIZE
