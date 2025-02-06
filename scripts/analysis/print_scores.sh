#!/bin/bash
#SBATCH --job-name=llm_la
#SBATCH --partition=lang_long
#SBATCH --account=lang
#SBATCH --cpus-per-task=80
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate llm-acceptability

DATA=blimp
# DATA=climp

# Running significant tests require a lot of CPUs, so use lang_* partition
PYTHONPATH=. python src/analysis/print_scores.py -d $DATA --show_percent -a sd \
  --significance_test --debug --summary_latex
