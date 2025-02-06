# This script is for explaining the strengths of each method in each phenomenon by the mean length difference between the good and bad sequences
# But the explanation was unsuccessful
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.constants import data_to_models, data_to_phenomenon_to_abbrev, model_to_abbrev
from src.utils import get_uid_to_phenomenon, load_logprobs_latest, load_result_latest

uid_to_phenomenon = get_uid_to_phenomenon()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="blimp")
    parser.add_argument(
        "--x", "-x", type=str, default="phenomenon", choices=["phenomenon", "paradigm"]
    )
    parser.add_argument(
        "--logprobs_dir",
        type=str,
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="results/blimp/lendiff_by_{x}.csv",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    approach = (
        "ll_intemplate_single"  # logprobs are saved in the directories of this approach
    )
    template = "template_blimp_single_1"  # Any template is fine
    phenomenon_to_abbrev = data_to_phenomenon_to_abbrev[args.data]
    dfs = []

    for model in data_to_models[f"{args.data}"]:
        result_dir = Path("results/blimp") / approach / model / template
        result = load_result_latest(result_dir)
        logprobs = load_logprobs_latest(result_dir, "whole_seq")

        x_to_lendiffs = defaultdict(list)

        for output, logprob_good, logprob_bad in zip(
            result["outputs"], logprobs["logprobs_good"], logprobs["logprobs_bad"]
        ):
            lendiff = len(logprob_good) - len(logprob_bad)
            # lendiff = abs(len(logprob_good) - len(logprob_bad))
            # ma = max(len(logprob_good), len(logprob_bad))
            # mi = min(len(logprob_good), len(logprob_bad))
            # lendiff = ma / mi

            if args.x == "phenomenon":
                key = phenomenon_to_abbrev[uid_to_phenomenon[output["UID"]]]
            elif args.x == "paradigm":
                key = output["UID"]

            x_to_lendiffs[key].append(lendiff)

        df = meanlendiff_by_x(x_to_lendiffs, args.x, model)
        dfs.append(df)

    df_all = pd.concat(dfs)
    out_file = args.out_file.format(x=args.x)
    df_all.to_csv(out_file, index=False)


def meanlendiff_by_x(x_to_lendiffs: dict, x: str, model: str) -> pd.DataFrame:
    phenomenon_to_meanlendiff = [
        {
            x: phenomenon,
            "lendiff": round(sum(lendiffs) / len(lendiffs), 3),
        }
        for phenomenon, lendiffs in x_to_lendiffs.items()
    ]
    df = pd.DataFrame(phenomenon_to_meanlendiff).sort_values(
        "lendiff", ascending=False
    )
    df["model"] = model_to_abbrev[model.split("_")[0]]
    # df["rank"] = range(1, len(df) + 1)
    # df = df[["model", "rank", x, "mean_len_diff"]]
    print(df)

    return df


if __name__ == "__main__":
    main()
