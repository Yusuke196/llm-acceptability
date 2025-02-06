#!/bin/bash
#SBATCH --job-name=llm_la
#SBATCH --partition=lang_short
#SBATCH --account=lang
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate llm-acceptability

for normalization in "normalized" "pen"; do
  for MODEL in "meta-llama/Meta-Llama-3-70B" "meta-llama/Meta-Llama-3-70B-Instruct"; do
  # for MODEL in "mistralai/Mixtral-8x7B-v0.1" "mistralai/Mixtral-8x7B-Instruct-v0.1"; do
  # for MODEL in "Qwen/Qwen2-57B-A14B" "Qwen/Qwen2-57B-A14B-Instruct"; do
    PYTHONPATH=. python src/pred_by_prob.py -d blimp \
      -a ll_${normalization} -m $MODEL -q "4bit" \
      --logprobs_dir results/blimp/ll
  done
done
