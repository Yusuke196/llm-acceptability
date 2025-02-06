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
  for i in {1..5}; do
    TEMPLATE=templates_ll/template_blimp_single_${i}.txt
    for MODEL in "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Meta-Llama-3-70B" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1" "Qwen/Qwen2-57B-A14B-Instruct" "Qwen/Qwen2-57B-A14B"; do
      PYTHONPATH=. python src/pred_by_prob.py -d blimp \
        -a ll_intemplate_single_${normalization} -t $TEMPLATE \
        -m $MODEL -q "4bit" \
        --logprobs_dir results/blimp/ll_intemplate_single
    done
  done
done
