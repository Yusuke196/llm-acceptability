import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr

from src.constants import approachname_to_officialname, data_to_models, model_to_abbrev
from src.utils import (
    add_iscorrect,
    get_result_dir,
    load_logprobs_latest,
    load_result_latest,
)

# Don't omit rows when printing DataFrames
pd.set_option("display.max_rows", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="blimp")
    parser.add_argument(
        "--approaches",
        "-a",
        type=str,
        nargs="+",
        default=[
            "ll",
            "ll_pen",
            "ll_intemplate_single",
            "ll_intemplate_single_pen",
            "prompting_yn",
        ],
    )
    parser.add_argument(
        "--approaches_chart",
        type=str,
        nargs="+",
        default=[
            "ll_intemplate_single",
            "ll_intemplate_single_pen",
            "prompting_yn",
        ],
    )
    parser.add_argument(
        "--logprobs_dir",
        type=str,
    )
    parser.add_argument(
        "--load_blimp_climp",
        action="store_true",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    if not args.load_blimp_climp:
        best_templates = pd.read_csv(
            f"results/{args.data}/best_templates.csv", index_col=0
        )

        records_all = []
        correlations = []

        for model in data_to_models[args.data]:
            # Get logprobs to compute lendiff
            intemplate_result_dir = get_result_dir(
                f"results/{args.data}",
                # Any template is fine
                f"ll_intemplate_single__template_{args.data}_single_1",
                model,
            )
            logprobs = load_logprobs_latest(intemplate_result_dir, "whole_seq")

            for approach in args.approaches:
                # Append template to approach if applicable
                if approach not in ["ll", "ll_pen"]:
                    model_abbrev = model_to_abbrev[model.split("_")[0]]
                    approach_template_str = best_templates.loc[approach, model_abbrev]
                else:
                    approach_template_str = approach

                result_dir = get_result_dir(
                    f"results/{args.data}", approach_template_str, model
                )
                result = load_result_latest(result_dir)
                outputs = add_iscorrect(result["outputs"])

                records = []
                for output, logprobs_good, logprobs_bad in zip(
                    outputs, logprobs["logprobs_good"], logprobs["logprobs_bad"]
                ):
                    records.append(
                        {
                            "lendiff": len(logprobs_good) - len(logprobs_bad),
                            "is_correct": output["is_correct"],
                        }
                    )

                records_all.extend(
                    {"model": model, "approach": approach, **r} for r in records
                )

                # Correlation
                corr, r = compute_corr(records)
                correlations.append(
                    {"model": model, "approach": approach, "corr": corr, "r": r}
                )

        df_all = pd.DataFrame(records_all)
        df_all["model"] = df_all["model"].str.replace("_4bit", "").map(model_to_abbrev)
        df_all["approach"] = df_all["approach"].map(approachname_to_officialname)

        df_all.to_csv(f"data/tmp/{args.data}_lendiff_iscorrect.csv", index=False)

        # Correlation
        df_corr_all = pd.DataFrame(correlations)
        print(df_corr_all)
        print()
        # Table
        df_corr = pd.DataFrame(df_corr_all.groupby("approach")["corr"].mean())
        df_corr.index = df_corr.index.map(approachname_to_officialname)
        s = df_corr.to_latex(float_format="{:.3f}".format)
        s = "    " + s.replace("\n", "\n    ")  # .replace("0.", ".")
        print(s)

        # Line chart
        # draw_linechart(df_all, args.approaches_chart, args.data)

    else:
        blimp = pd.read_csv("data/tmp/blimp_lendiff_iscorrect.csv")
        climp = pd.read_csv("data/tmp/climp_lendiff_iscorrect.csv")
        climp["model"] = " " + climp["model"] + " "
        df_all = pd.concat([blimp, climp])

        # Line chart
        draw_linechart(df_all, args.approaches_chart, "blimp_climp")


def compute_corr(records: list[int, list[bool]]):
    lendiffs = [r["lendiff"] for r in records]
    iscorrects = [r["is_correct"] for r in records]

    r, pvalue = pointbiserialr(lendiffs, iscorrects)
    corr = np.corrcoef(lendiffs, iscorrects)[0, 1]
    assert round(corr, 6) == round(r, 6)

    return r, pvalue


def draw_linechart(df: pd.DataFrame, approaches: list[str], data_name: str) -> None:
    approaches_official = [approachname_to_officialname[k] for k in approaches]
    df = df[df["approach"].isin(approaches_official)].copy()

    # df = delete_rows_of_rare_lendiffs(df)

    df["facet_order"] = df["model"].map(
        {
            "Llama-3": 1,
            "Mixtral": 2,
            "Qwen2": 3,
            " Llama-3 ": 3.1,
            " Yi-1.5 ": 3.2,
            " Qwen2 ": 3.3,
            "Llama-3-Instruct": 4,
            "Mixtral-Instruct": 5,
            "Qwen2-Instruct": 6,
            " Llama-3-Instruct ": 6.1,
            " Yi-1.5-Chat ": 6.2,
            " Qwen2-Instruct ": 6.3,
        }
    )
    df.sort_values("facet_order", inplace=True)

    if data_name == "blimp":
        n_cols = 3
    if data_name == "climp":
        n_cols = 2
    elif data_name == "blimp_climp":
        n_cols = 6

    g = sns.FacetGrid(df, col="model", hue="approach", col_wrap=n_cols)
    g.map_dataframe(sns.lineplot, x="lendiff", y="is_correct")
    g.set_titles("{col_name}")
    g.set(
        xlabel="Token-length diff.",
        xticks=[-5, 0, 5],
        ylabel="Accuracy",
    )
    g.add_legend(label_order=approaches_official)
    sns.move_legend(
        g,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(approaches_official),
        title=None,
        frameon=False,
    )
    g.figure.set_size_inches(9.2, 3)
    g.figure.subplots_adjust(
        wspace=0.13, hspace=0.42, left=0.05, right=0.97, top=0.83, bottom=0.17
    )
    if data_name == "blimp_climp":
        for i, ax in enumerate(g.axes.flat):
            # If the subplot is in the fourth, fifth, or sixth column
            if i % n_cols in [3, 4, 5]:
                pos = ax.get_position()  # get the original position
                pos.x0 += 0.015  # Move the left edge to the right
                pos.x1 += 0.015  # Move the right edge to the right
                ax.set_position(pos)

    plt.savefig(f"results/{data_name}/accuracy_by_lendiff.pdf")


# def delete_rows_of_rare_lendiffs(df: pd.DataFrame, threshold=10) -> pd.DataFrame:
#     has_enough_sample = (
#         df.groupby(["model", "approach", "lendiff"]).count() >= threshold
#     )


if __name__ == "__main__":
    main()
