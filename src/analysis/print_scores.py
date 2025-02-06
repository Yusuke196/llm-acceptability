import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from src.constants import (
    approachname_to_officialname,
    data_to_approaches,
    data_to_models,
    model_to_abbrev,
)
from src.utils import format_df, get_result_dir, judge_correctness, load_result_latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="blimp")
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--main_stat", "-m", type=str, default="mean")
    parser.add_argument("--show_percent", "-p", action="store_true")
    parser.add_argument(
        "--add_stats_in_summary",
        "-a",
        type=str,
        default=[],
        choices=["sd", "max", "min-max"],
    )
    parser.add_argument(
        "--summary_path", type=str, default="results/{data}/summary_{main_stat}.csv"
    )
    parser.add_argument(
        "--max_path", type=str, default="results/{data}/max.csv"
    )  # Used in majority_vote.py
    parser.add_argument("--all_path", type=str, default="results/{data}/all.csv")
    parser.add_argument("--load_df_all", action="store_true")
    parser.add_argument("--significance_test", action="store_true")
    parser.add_argument(
        "--best_templates_path", type=str, default="results/{data}/best_templates.csv"
    )
    parser.add_argument(
        "--swarmplot_path", type=str, default="results/{data}/swarmplot.pdf"
    )
    parser.add_argument("--summary_latex", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()
    in_dir = f"results/{args.data}"
    approaches = data_to_approaches[args.data]
    models = data_to_models[args.data]

    if not args.load_df_all:
        # Column: model, row: approach
        df_all = get_overall_results(
            in_dir,
            approaches,
            models,
            args.metric,
            abbreviate_col_names=True,
        )
        if args.all_path:
            all_path = args.all_path.format(data=args.data)
            df_all.to_csv(all_path)
    else:
        all_path = args.all_path.format(data=args.data)
        df_all = pd.read_csv(all_path, index_col=0)

    print(df_all)
    print()

    if args.significance_test:
        test_result_df = perform_significance_tests(
            in_dir, approaches, models, debug=args.debug
        )[df_all.columns]
        pd.set_option("display.max_columns", None)
        print("P-values:")
        print(test_result_df)
        print()

        test_result_df_aster = convert_pvalues_to_symbols(test_result_df)

    # Not used in the paper
    # draw_swarmplot(df_all.copy(), args.swarmplot_path.format(data=args.data))

    df_summary = summarize_results(
        df_all,
        args.main_stat,
        args.add_stats_in_summary,
        args.show_percent,
        mark_highest=True,
        mark_second=True,
        significance_df=test_result_df_aster if args.significance_test else None,
        debug=args.debug,
    )
    print("Summary:")
    print(df_summary)
    print()
    if args.summary_path:
        df_summary.to_csv(
            args.summary_path.format(data=args.data, main_stat=args.main_stat)
        )

    # Save the maximum scores for majority vote
    if args.max_path:
        df_max = get_max(df_all)
        df_max.to_csv(args.max_path.format(data=args.data))

    # Save best templates for the radar chart, etc.
    if args.best_templates_path:
        df_best_templates = get_best_templates(df_all)
        df_best_templates.to_csv(args.best_templates_path.format(data=args.data))

    # Print summary in LaTeX
    if args.summary_latex:
        s = df_summary.to_latex(float_format="{:.3f}".format)
        s = "    " + s.replace("\n", "\n    ")
        print(s)
        print()


def get_overall_results(
    in_dir: str,
    approaches: list[str],
    models: list[str],
    metric: list[str],
    abbreviate_col_names: bool,
) -> pd.DataFrame:
    def get_row(approach: str) -> dict:
        row = {}
        for model in models:
            result_dir = get_result_dir(in_dir, approach, model)
            result = load_result_latest(result_dir)
            model_s = model.split("_")[0]  # Remove quantization info

            if result is not None:
                row[model_s] = result["scores"][metric]
            else:
                row[model_s] = None

        # This looks like {"ll": 0.5, "ll_normalized": 0.6, ... }
        return row

    parallel = Parallel(-1)
    rows = parallel(delayed(get_row)(approach) for approach in approaches)
    df = pd.DataFrame(rows, index=approaches)

    if abbreviate_col_names:
        df.columns = [model_to_abbrev[model] for model in df.columns]

    return df


def get_result_list_df(
    in_dir: str,
    approaches: list[str],
    models: list[str],
    abbreviate_col_names: bool,
) -> pd.DataFrame:
    """
    Return a DataFrame where each cell is a list of 0s and 1s.
    """

    def get_row(approach: str) -> dict:
        row = {}
        for model in models:
            result_dir = get_result_dir(in_dir, approach, model)
            result = load_result_latest(result_dir)
            model_s = model.split("_")[0]  # Remove quantization info

            if result is not None:
                row[model_s] = [
                    1 if judge_correctness(output) else 0
                    for output in result["outputs"]
                ]
            else:
                row[model_s] = None

        # This looks like {"ll": [1, 0, ...], "ll_normalized": [0, 1, ...], ...}
        return row

    parallel = Parallel(-1)
    rows = parallel(delayed(get_row)(approach) for approach in approaches)
    df = pd.DataFrame(rows, index=approaches)

    if abbreviate_col_names:
        df.columns = [model_to_abbrev[model] for model in df.columns]

    return df


def perform_significance_tests(
    in_dir: str,
    approaches: list[str],
    models: list[str],
    debug: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame where each cell contains a p-value.
    """

    print("Performing significance tests...")
    methods = [
        "ll_intemplate_single",
        "ll_intemplate_single_normalized",
        "ll_intemplate_single_pen",
        "ll_intemplate_compar",
        "prompting",
        "prompting_yn",
    ]
    n_iterations = 1000

    if debug:
        # Comment in for faster debugging
        # methods = ["ll_intemplate_single"]
        # approaches = approaches[:8]
        n_iterations = 3

    # Column: model, row: approach
    result_list_df = get_result_list_df(
        in_dir, approaches, models, abbreviate_col_names=True
    )

    res = []
    for model in result_list_df.columns:
        print("Model:", model)
        for method in tqdm(methods):
            # print("Method:", method)
            result_lists: pd.Series = result_list_df[model].dropna()
            p_value = _perform_bootstrap_test(result_lists, method, n_iterations)
            res.append(
                {
                    "model": model,
                    "method": method,
                    "p_value": p_value,
                }
            )

        print()

    df = pd.DataFrame(res).pivot_table(
        index="method", columns="model", values="p_value"
    )

    return df


def _perform_bootstrap_test(
    result_lists: pd.Series, method: str, n_iterations: int
) -> float:
    """
    Perform a one-sided test where the null hypothesis is that `method` is not better than ll.
    """

    def get_bootstrap_diff() -> float:
        result_lists_target: pd.Series = result_lists[
            result_lists.index.str.startswith(method + "__")
        ]
        # Convert the Series to more convenient format
        # Column: template, row: minimal pair
        result_df_target = pd.DataFrame(
            result_lists_target.tolist(), index=result_lists_target.index
        ).T
        # Each element corresponds to a minimal pair
        result_list_ll = result_lists.loc["ll"]

        sampled_indices = random.choices(
            range(len(result_list_ll)), k=len(result_list_ll)
        )
        # print(f'{sampled_indices[:10] = }')
        # sampled_result_df_target.shape == (n_minimal_pairs, n_templates)
        sampled_result_df_target = result_df_target.iloc[sampled_indices]
        # len(result_list_ll) == n_minimal_pairs
        sampled_results_ll = [result_list_ll[i] for i in sampled_indices]

        mean_target = np.mean(sampled_result_df_target.values.flatten().tolist())
        mean_ll = np.mean(sampled_results_ll)
        return mean_target - mean_ll

    parallel = Parallel(-1)
    bootstrap_diffs = parallel(
        delayed(get_bootstrap_diff)() for _ in range(n_iterations)
    )
    p_value = np.mean([diff <= 0 for diff in bootstrap_diffs])

    return p_value


def convert_pvalues_to_symbols(test_result_df: pd.DataFrame) -> pd.DataFrame:
    def func(p_value: float) -> str:
        if p_value < 0.01:
            return "$^{**}$"
        elif p_value < 0.05:
            return "$^*$"
        else:
            return ""

    test_result_df_aster = test_result_df.map(func)

    # Put empty strings in rows for conventional methods because we don't perform tests for them
    for method in ["ll", "ll_normalized", "ll_pen"]:
        test_result_df_aster.loc[method] = ""

    return test_result_df_aster


def summarize_results(
    df_all: pd.DataFrame,
    main_stat: str,
    add_stats_in_summary: list[str] = [],
    show_percent: bool = False,
    mark_highest: bool = False,
    mark_second: bool = False,
    significance_df: pd.DataFrame = None,
    debug: bool = False,
    sort_rows: bool = True,
) -> pd.DataFrame:
    if main_stat == "mean":
        df = df_all.groupby(df_all.index.str.split("__").str[0]).mean()
    elif main_stat == "max":
        df = df_all.groupby(df_all.index.str.split("__").str[0]).max()

    df = format_df(
        df,
        show_percent=show_percent,
        mark_highest=mark_highest,
        mark_second=mark_second,
    )

    if significance_df is not None:
        assert df.columns.equals(significance_df.columns)
        df = df + significance_df

    if "sd" in add_stats_in_summary:
        df_stddev = df_all.groupby(df_all.index.str.split("__").str[0]).std()
        # Concatenate two strings for each cell
        df = df + _format_stddev(df_stddev, show_percent)

        # mean_stddev_over_models = df_stddev.mean(axis=1).dropna().round(3)
        # print(f"Mean of stddev:")
        # print(mean_stddev_over_models)
        # print()

    if "max" in add_stats_in_summary:
        df_max = df_all.groupby(df_all.index.str.split("__").str[0]).max()
        df_max = format_df(
            df_max,
            show_percent=show_percent,
            mark_highest=mark_highest,
            mark_second=mark_second,
        )
        df = df + "/" + df_max

    if "min-max" in add_stats_in_summary:
        df_max = (
            df_all.groupby(df_all.index.str.split("__").str[0])
            .max()
            .map(lambda x: f"{x:.3f}".replace("0.", "."))
        )
        df_min = (
            df_all.groupby(df_all.index.str.split("__").str[0])
            .min()
            .map(lambda x: f"{x:.3f}".replace("0.", "."))
        )
        df = df + _get_min_max_df(df_min, df_max)

    df.rename(index=approachname_to_officialname, inplace=True)
    # When debug is True, sorting results in KeyError because some rows are missing
    # if debug:
    #     sort_rows = False

    if sort_rows:
        df_sorted = df.loc[approachname_to_officialname.values()]
    else:
        df_sorted = df

    return df_sorted


# def _format_stddev(value: float):
#     # Needs to return "" for approaches that don't use templates because they don't have stddev
#     if np.isnan(value):
#         return ""

#     return f"±{value:.3f}"


def _format_stddev(df_stddev: pd.DataFrame, show_percent: bool) -> pd.DataFrame:
    def func(x: float) -> str:
        if np.isnan(x):
            return ""

        if show_percent:
            return "$_{\pm " + f"{x * 100:.1f}" + "}$"
        else:
            return f"±{x:.3f}"

    return df_stddev.map(func)


def _get_min_max_df(df_min: pd.DataFrame, df_max: pd.DataFrame) -> pd.DataFrame:
    df = " (" + df_min + "-" + df_max + ")"
    for idx in ["ll", "ll_normalized", "ll_pen"]:
        df.loc[idx] = ""

    return df


def get_max(df_all: pd.DataFrame) -> pd.DataFrame:
    df_max = (
        df_all.groupby(df_all.index.str.split("__").str[0])
        .max()
        .map(lambda x: f"{x:.3f}")
    )
    df_max.index = df_max.index.map(approachname_to_officialname)
    df_max_sorted = df_max.loc[approachname_to_officialname.values()]

    return df_max_sorted


def get_best_templates(df: pd.DataFrame) -> pd.DataFrame:
    approaches = [
        a
        for a in approachname_to_officialname.keys()
        # These approaches don't have templates
        if (a not in ["ll", "ll_normalized", "ll_pen"])
    ]
    dct = {}

    for approach in approaches:
        df_sub = df.query(f'index.str.startswith("{approach}__")')
        # print(df_sub)
        dct[approach] = df_sub.idxmax(axis=0)

    return pd.DataFrame(dct).T


def draw_swarmplot(df: pd.DataFrame, swarmplot_path: str) -> None:
    # Tidy the data
    df["approach"] = df.index.str.split("__").str[0].map(approachname_to_officialname)
    approaches = [
        "Prompting",
        "PenLP",
        "In-template LP",
        "In-template comparative LP",
    ]
    df = df[df["approach"].isin(approaches)]
    # Convert the wide format to the long format
    df = df.melt(id_vars="approach", var_name="model", value_name="score")
    df = df.dropna()

    # Order the facets (necessary only when arranging facets in two rows)
    df["facet_order"] = df["model"].map(
        {
            "Llama-3": 1,
            "Mixtral": 2,
            "Qwen2": 3,
            "Llama-3-Instruct": 4,
            "Mixtral-Instruct": 5,
            "Qwen2-Chat": 6,
        }
    )
    df.sort_values("facet_order", inplace=True)
    # print(df)

    # Use FacetGrid because it is an organized way to draw multiple subplots
    # Don't set `height` and `aspect` here, and fit them to the entire figure later
    g = sns.FacetGrid(df, col="model", hue="approach", col_wrap=3)
    g.map_dataframe(sns.swarmplot, x="approach", y="score", order=approaches)

    # Mark averages
    means = df.groupby(["model", "approach"]).mean().reset_index()
    for ax, model in zip(g.axes.flat, df["model"].unique()):
        means_2 = (
            means.query(f'model == "{model}"').set_index("approach").loc[approaches]
        )
        ax.scatter(
            range(len(approaches)),
            means_2["score"],
            marker="_",
            linewidths=1,  # Make the marker thinner
            s=80,
            color="#222222",
            zorder=10,  # Draw on top of the swarmplot
        )
        # Mark the chance rate
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.6)

    # Adjust the appearance
    g.set(xlabel=None, xticklabels=[], ylabel="Accuracy", ylim=(0.44, 0.96))
    # g.refline(y=0.5, linewidth=0.4)
    g.set_titles("{col_name}")
    g.add_legend(label_order=approaches)
    # See https://seaborn.pydata.org/generated/seaborn.move_legend.html for the arguments
    sns.move_legend(
        g,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(approaches),
        title=None,
        frameon=False,
    )
    # Note: In the ACL format, the width of the two columns and in-between space combined is 16 cm (approx. 6.3 inch)
    # https://acl-org.github.io/ACLPUB/formatting.html
    g.figure.set_size_inches(8.1, 5.4)
    # Adjust the spaces between the plots by setting `wspace` and `hspace`
    # Fit the subplots to the entire figure by setting `left` and `right`. The left margin should be larger because there are labels
    g.figure.subplots_adjust(wspace=0.12, hspace=0.24, left=0.08, right=0.98)

    plt.savefig(swarmplot_path)


if __name__ == "__main__":
    main()
