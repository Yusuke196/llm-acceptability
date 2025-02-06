#!/bin/bash -eu

TEMPLATE_TYPE=single
# TEMPLATE_TYPE=compar

# rm -rf ~/.cache/huggingface/datasets/generator/

APPROACH=ll_intemplate_${TEMPLATE_TYPE}

# Run relatively large models with a smaller batch size
for i in {1..5}; do
  TEMPLATE=templates_ll/template_blimp_${TEMPLATE_TYPE}_${i}.txt

  # for MODEL in "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Meta-Llama-3-70B"; do
	# 	sbatch scripts/_ll_intemplate_blimp.sh $APPROACH $TEMPLATE $MODEL 32
	# done

  # for MODEL in "Qwen/Qwen2-57B-A14B-Instruct" "Qwen/Qwen2-57B-A14B"; do
  for MODEL in "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1"; do
    sbatch scripts/_ll_intemplate_blimp.sh $APPROACH $TEMPLATE $MODEL 64
	done
done

# # For debugging, run the following on gpu_intr
# PYTHONPATH=. python src/pred_by_prob.py -d blimp \
#   -a $APPROACH -t templates_ll/template_blimp_${TEMPLATE_TYPE}_1.txt \
#   -m "meta-llama/Meta-Llama-3-8B" -q "4bit" \
#   -b 64 --debug
