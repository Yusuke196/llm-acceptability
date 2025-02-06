#!/bin/bash -eu

TEMPLATE_TYPE=single
# TEMPLATE_TYPE=compar

# rm -rf ~/.cache/huggingface/datasets/generator/

for i in {1..5}; do
  TEMPLATE=templates_ll/template_climp_${TEMPLATE_TYPE}_${i}.txt
  # for MODEL in "Qwen/Qwen2-57B-A14B-Instruct" "Qwen/Qwen2-57B-A14B"; do
  # for MODEL in "meta-llama/Meta-Llama-3-70B" "meta-llama/Meta-Llama-3-70B-Instruct"; do
  for MODEL in "01-ai/Yi-1.5-34B" "01-ai/Yi-1.5-34B-Chat"; do
    sbatch scripts/_ll_intemplate_climp.sh \
      ll_intemplate_${TEMPLATE_TYPE} $TEMPLATE $MODEL 32
	done
done

# For debugging, run the following on a node with a GPU
# TEMPLATE_TYPE=single; PYTHONPATH=. python src/pred_by_prob.py -d climp \
#   -a ll_intemplate_${TEMPLATE_TYPE} -t templates_ll/template_climp_${TEMPLATE_TYPE}_1.txt \
#   -m "Qwen/Qwen1.5-0.5B" -q "4bit" \
#   -b 64 --debug
