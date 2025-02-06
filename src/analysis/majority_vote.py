import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.constants import data_to_models, model_to_abbrev
from src.utils import (
    format_df,
    get_result_dir,
    get_uid_to_phenomenon,
    judge_correctness,
    load_result_latest,
)

modelabbrev_to_officialname = {v: k for k, v in model_to_abbrev.items()}
uid_to_phenomenon = get_uid_to_phenomenon()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="blimp")
    parser.add_argument(
        "--random", "-r", action="store_true", help="Randomly select templates"
    )
    parser.add_argument("--iter_count", "-i", type=int, default=10)
    parser.add_argument("--show_percent", "-p", action="store_true")
    parser.add_argument(
        "--out_file", "-o", type=str, default="results/{data}/majority_vote.csv"
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    df_all = pd.read_csv(f"results/{args.data}/all.csv", index_col=0)

    # Majority voting
    df_vote = get_df_vote(df_all, args.data, args.random, args.iter_count)

    # Get max scores
    df_max = pd.read_csv(f"results/{args.data}/max.csv", index_col=0).loc[
        ["In-template LP", "Yes/No prob comp"]
    ]
    df_max.index = df_max.index.map(lambda x: x + " (oracle)")

    assert df_vote.columns.equals(df_max.columns)
    df_concated = pd.concat([df_vote, df_max], axis=0)
    formatted = format_df(df_concated, show_percent=True, mark_highest=True)

    out_file = args.out_file.format(data=args.data)
    formatted.to_csv(out_file)

    s = formatted.to_latex()
    s = "    " + s.replace("\n", "\n    ")
    print(s)


def get_df_vote(
    df_all: pd.DataFrame, data_name: str, random: bool, iter_count: int
) -> pd.DataFrame:
    res = []

    # This list includes strings like "Qwen2-57B-A14B_4bit"
    models = data_to_models[data_name]

    for model in models:
        model_abb = model_to_abbrev[model.split("_")[0]]

        for prompt_count in [5, 3, 2, 0]:
            print(f"model: {model}, prompt_count: {prompt_count}")
            intemp_count = 5 - prompt_count
            # accuracies = []

            def func() -> pd.Series:
                approach_template_strs = get_approach_template_strs(
                    df_all, model_abb, prompt_count, intemp_count, random=random
                )

                # Load results
                outputs_list = []
                for approach_template_str in approach_template_strs:
                    result_dir = get_result_dir(
                        f"results/{data_name}",
                        approach_template_str,
                        model,
                    )
                    outputs = load_result_latest(result_dir)["outputs"]
                    outputs_list.append(outputs)

                assert len(outputs_list) == 5
                if data_name == "blimp":
                    assert len(outputs_list[0]) == 67000

                # Majority voting
                # majority_voting_is_correct = []
                dcts = []
                for records in zip(*outputs_list):
                    flags = [judge_correctness(record) for record in records]
                    # majority_voting_is_correct.append(sum(flags) >= 3)
                    if data_name == "blimp":
                        phenom = uid_to_phenomenon[records[0]["UID"]]
                    elif data_name == "climp":
                        phenom = records[0]["phenomenon"]

                    dcts.append(
                        {
                            "phenom": phenom,
                            "is_correct": sum(flags) >= 3,
                        }
                    )

                # accuracy = np.mean(majority_voting_is_correct)
                df = pd.DataFrame(dcts)
                accuracies = df.groupby("phenom")["is_correct"].mean()
                accuracies["overall"] = df["is_correct"].mean()

                return accuracies

            parallel = Parallel(n_jobs=-1)
            accuracies_list = parallel(delayed(func)() for _ in tqdm(range(iter_count)))
            # Accuracies averaged over iterations
            accuracies_by_phenom = pd.concat(accuracies_list, axis=1).mean(axis=1)

            res.append(
                {
                    "model": model_abb,
                    "approach": f"Ensemble L{intemp_count}:P{prompt_count}",
                    "accuracy": accuracies_by_phenom["overall"],
                }
            )

            # Print accuracies of all phenomena to see if some don't reach human level
            print(accuracies_by_phenom)

    df = pd.DataFrame(res)
    df["accuracy"] = df["accuracy"]
    df_2 = df.pivot(index="approach", columns="model", values="accuracy")[
        df_all.columns
    ]

    return df_2


def get_approach_template_strs(
    df_all: pd.DataFrame, model: str, prompt_count: int, intemp_count: int, random: bool
) -> list[str]:
    strs_prompt = take_k_templates(
        df_all,
        model,
        approach="prompting_yn",
        k=prompt_count,
        random=random,
    )

    strs_intemp = take_k_templates(
        df_all,
        model,
        approach="ll_intemplate_single",
        k=intemp_count,
        random=random,
    )

    return strs_prompt + strs_intemp


k_to_indices = {
    0: [],
    2: [1, 3],
    3: [1, 2, 3],
    5: [0, 1, 2, 3, 4],
}


def take_k_templates(
    df_all: pd.DataFrame, model: str, approach: str, k: int, random: bool
) -> list[str]:
    df_all_2 = df_all[df_all.index.str.startswith(approach + "__")]
    srs = df_all_2[model].dropna()
    if random:
        approach_template_strs = srs.sample(k).index
    else:
        approach_template_strs = srs.sort_values(ascending=False).index[k_to_indices[k]]

    return list(approach_template_strs)


if __name__ == "__main__":
    main()
