# Note: This script uses spacy, taking a lot of time
import argparse
import json
from collections import Counter

import pandas as pd
import spacy
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    outputs = json.load(
        open(
            # Any result file is fine
            "results/blimp/ll_intemplate_single/"
            "Meta-Llama-3-70B_4bit/template_blimp_single_1/20240530-144714.json"
        )
    )["outputs"]

    nlp = spacy.load("en_core_web_lg")

    res = []
    for output in tqdm(outputs):
        words_good = [token.text.lower() for token in nlp(output["sentence_good"])]
        counter_good = Counter(words_good)
        words_bad = [token.text.lower() for token in nlp(output["sentence_bad"])]
        counter_bad = Counter(words_bad)
        res.append(
            {"paradigm": output["UID"], "bowmatches": counter_good == counter_bad}
        )

    df = pd.DataFrame(res)
    table = pd.pivot_table(df, index="paradigm", values="bowmatches", aggfunc="mean")
    table.rename(columns={"bowmatches": "bowmatchratio"}, inplace=True)
    table.to_csv("data/blimp/paradigm_to_bowmatchratio.csv")


if __name__ == "__main__":
    main()
