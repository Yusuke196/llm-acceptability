# Set up environment

```
pip install -r requirements.txt
```

# Download data

```
bash scripts/download_data.sh
```

# Perform experiments

Sentence probability readout
```
sbatch scripts/ll_blimp.sh
sbatch scripts/ll_climp.sh
sbatch scripts/ll_norm_blimp.sh
sbatch scripts/ll_norm_climp.sh
```

In-template probability readout
```
sbatch scripts/ll_intemplate_blimp.sh
sbatch scripts/ll_intemplate_climp.sh
sbatch scripts/ll_intemplate_norm_blimp.sh
sbatch scripts/ll_intemplate_norm_climp.sh
```

Prompt-based methods
```
sbatch scripts/prompting_blimp.sh
sbatch scripts/prompting_climp.sh
sbatch scripts/prompting_yn_blimp.sh
sbatch scripts/prompting_yn_climp.sh
```

# Generate tables and figures

tab:summary-mean
```
sbatch scripts/analysis/print_scores.sh
```

fig:accuracy-by-lendiff, tab:success-lendiff-correlation
```
bash scripts/analysis/accuracy_by_lendiff.sh
```

fig:accuracy-by-phenomenon
```
PYTHONPATH=. python src/analysis/accuracy_by_x.py -x phenomenon -d blimp
PYTHONPATH=. python src/analysis/accuracy_by_x.py -x phenomenon -d climp
```

tab:ensemble
```
PYTHONPATH=. python src/analysis/majority_vote.py -d blimp --show_percent
PYTHONPATH=. python src/analysis/majority_vote.py -d climp --show_percent
PYTHONPATH=. python src/paper_writing/combine_tables.py -b results/blimp/majority_vote.csv -c results/climp/majority_vote.csv -d "Mixtral" "Mixtral-Instruct" "Yi-1.5" "Yi-1.5-Chat"
```

tab:summary-max
```
PYTHONPATH=. python src/analysis/print_scores.py -d blimp --show_percent \
    --main_stat max --summary_latex
PYTHONPATH=. python src/analysis/print_scores.py -d climp --show_percent \
    --main_stat max --summary_latex
```

tab:a-proportion
```
PYTHONPATH=. python src/analysis/compute_ab_ratio.py
PYTHONPATH=. python src/analysis/compute_ab_ratio.py -d climp
```
