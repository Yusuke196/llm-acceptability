#!/bin/bash -eu

export PYTHONPATH=.
python src/analysis/accuracy_by_lendiff.py -d blimp
python src/analysis/accuracy_by_lendiff.py -d climp
python src/analysis/accuracy_by_lendiff.py --load_blimp_climp
