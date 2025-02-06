import argparse
import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file",
        "-i",
        type=str,
        default="results/blimp/accdiff_by_paradigm.csv",
    )
    parser.add_argument(
        "--in_bow_file", type=str, default="data/blimp/paradigm_to_bowmatchratio.csv"
    )
    parser.add_argument(
        "--out_pdf", "-o", type=str, default="results/blimp/accuracy_by_bowmatches.pdf"
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.in_file)
    df_bow = pd.read_csv(args.in_bow_file)
    df_bow["type"] = df_bow["bowmatchratio"].map(
        lambda x: "Word-shuffling" if x == 1 else "Other"
    )

    df_merged = df.merge(
        df_bow,
        left_on="paradigm",
        right_on="paradigm",
        how="left",
    )
    df_long = df_merged.melt(
        id_vars=["model", "type", "paradigm"],
        value_vars=["ll_intemplate_single", "prompting_yn"],
        var_name="method",
        value_name="acc",
    )
    print(df_long)
    print()

    # Accuracy averaged over models and methods
    df_mean = df_long.pivot_table(
        index=["model", "type", "method"],
        aggfunc="mean",
        values="acc",
    )
    print("Mean by model, type, and method:")
    print(df_mean.round(3))
    print()

    draw_swarmplot(df_long, args.out_pdf)

    df_mean_2 = df_long.pivot_table(
        index="paradigm", aggfunc="mean", values="acc"
    ).sort_values("acc", ascending=False)
    df_mean_by_paradigm = pd.merge(
        df_mean_2, df_bow, left_index=True, right_on="paradigm"
    ).drop(columns=["bowmatchratio"])
    print("Mean by paradigm:")
    pd.set_option("display.max_rows", None)
    print(df_mean_by_paradigm.round(3))
    print()

    # Accuracy of humans
    df_human = pd.read_csv(
        "data/blimp/raw_results/summary/human_validation_summary.csv"
    ).merge(
        df_bow,
        left_on="Condition",
        right_on="paradigm",
        how="left",
    )
    df_human_2 = df_human.pivot_table(index="type", aggfunc="mean", values="total_mean")
    print("Humans:")
    print(df_human_2.round(3))


def draw_swarmplot(df_long: pd.DataFrame, out_pdf: str) -> None:
    # Sort by models and attractor types
    df_long["model_order"] = df_long["model"].map(
        {
            "Llama-3": 1,
            "Mixtral": 2,
            "Qwen2": 3,
            "Llama-3-Instruct": 4,
            "Mixtral-Instruct": 5,
            "Qwen2-Instruct": 6,
        }
    )
    df_long["type_order"] = df_long["type"].map(
        {
            "Word-shuffling": 1,
            "Other": 2,
        }
    )
    df_long.sort_values(["model_order", "type_order"], inplace=True)
    df_long["method"] = df_long["method"].map(
        {"ll_intemplate_single": "In-template LP", "prompting_yn": "Yes/No prob comp"}
    )

    # Draw the swarmplot
    palette = "colorblind"
    g = sns.FacetGrid(
        df_long, col="model", row="method", hue="type", margin_titles=True
    )
    g.refline(y=0.5, linewidth=0.4)  # Mark the chance accuracy
    g.map_dataframe(sns.swarmplot, x="type", y="acc", s=2.3)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set(xlabel="", ylabel="Accuracy", xlim=[-0.4, 1.6])
    g.set_xticklabels(["W-shuf.", "Other"])
    sns.set_palette(palette)
    g.add_legend()
    sns.move_legend(
        g,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        title=None,
        frameon=False,
    )

    # Append human scores
    human_valid_summary = pd.read_csv(
        "data/blimp/raw_results/summary/human_validation_summary.csv"
    )
    df_long_2 = df_long.merge(
        human_valid_summary,
        left_on="paradigm",
        right_on="Condition",
        how="left",
    ).rename(columns={"total_mean": "human_acc"})

    # Prepare averages
    df_mean = df_long_2.pivot_table(
        index=["model", "method", "type"],
        values=["acc", "human_acc"],
        aggfunc="mean",
    ).reset_index()
    df_mean["type_order"] = df_mean["type"].map(
        {
            "Word-shuffling": 1,
            "Other": 2,
        }
    )
    type_len = len(df_long_2["type"].unique())

    # Mark the averages and human scores
    product_list = list(
        itertools.product(df_long_2["method"].unique(), df_long_2["model"].unique())
    )
    for ax, (method, model) in zip(g.axes.flat, product_list):
        assert ax.title._text in [model, ""], f"{ax.title._text} != {model}"

        df_mean_2 = df_mean.query(
            f'model == "{model}" and method == "{method}"'
        ).sort_values("type_order")

        # Mark the average
        ax.scatter(
            range(type_len),
            df_mean_2["acc"],
            marker="_",
            linewidths=1,  # Make the marker thinner
            s=60,
            color="#444444",
            # color=sns.color_palette(palette)[0],
            zorder=10,  # Put the mark on top of the swarmplot
        )
        # Mark the human score
        ax.scatter(
            range(type_len),
            df_mean_2["human_acc"],
            marker="x",
            s=16,
            color="#777777",
            zorder=9,
        )

    # Set the figure size
    g.figure.set_size_inches(10, 3.3)
    g.figure.subplots_adjust(top=0.86, bottom=0.07, left=0.07, right=0.97)

    plt.savefig(out_pdf)


if __name__ == "__main__":
    main()
