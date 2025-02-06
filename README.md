# Scripts for BLiMP experiments

Sentence probability readout
```
ll_blimp.sh
ll_norm_blimp.sh
```

In-template probability readout
```
ll_intemplate_blimp.sh
ll_intemplate_norm_blimp.sh
```

Prompting
```
prompting_blimp.sh
prompting_yn_blimp.sh
```

Note
- Replace `blimp` with `climp` for CLiMP experiments

# Scripts for tables and figures

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
