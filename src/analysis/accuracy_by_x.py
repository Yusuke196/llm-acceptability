# Run src/write_uid_to_phenomenon.py and src/print_scores.py before this script
import argparse
import json
import pickle
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from plotly.subplots import make_subplots

from src.constants import (
    approachname_to_officialname,
    data_to_models,
    data_to_modelssorted,
    data_to_phenomenon_to_abbrev,
    model_to_abbrev,
)
from src.utils import (
    add_iscorrect,
    get_result_dir,
    get_uid_to_phenomenon,
    load_result_latest,
)

uid_to_phenomenon = get_uid_to_phenomenon()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="blimp")
    parser.add_argument(
        "--x", "-x", type=str, default="phenomenon", choices=["phenomenon", "paradigm"]
    )
    parser.add_argument("--load_blimp_climp", action="store_true")
    parser.add_argument("--latex", "-l", action="store_true")
    parser.add_argument(
        "--figure_layout", type=str, default="2x6", choices=["3x4", "2x6"]
    )
    parser.add_argument("--no_figure_update", action="store_true")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    # `approaches` will correspond to traces in each subplot
    approaches = ["ll_intemplate_single", "prompting_yn"]
    best_templates = pd.read_csv(f"results/{args.data}/best_templates.csv", index_col=0)
    if args.figure_layout == "3x4":
        models = data_to_models[f"{args.data}"]
    elif args.figure_layout == "2x6":
        models = data_to_modelssorted[f"{args.data}"]

    if args.x == "phenomenon":
        if not args.load_blimp_climp:
            phenomenon_to_abbrev = data_to_phenomenon_to_abbrev[args.data]

            model_to_dfphenomenon = {}  # dfphenomenon: approach x phenomenon

            # `models` will correspond to subplots (radar charts)
            for model in models:
                df_phenomenon = accuracy_by(
                    args.data, model, approaches, best_templates, "phenomenon"
                )
                df_phenomenon.columns = [
                    phenomenon_to_abbrev[p] for p in df_phenomenon.columns
                ]
                # Sort columns
                df_phenomenon = df_phenomenon[phenomenon_to_abbrev.values()]
                print(df_phenomenon)
                print()

                model_abbrev = model_to_abbrev[model.split("_")[0]]
                model_to_dfphenomenon[model_abbrev] = df_phenomenon

            dfphenomenon_list = list(model_to_dfphenomenon.values())
            df_concated = concat_dfs(dfphenomenon_list)
            count_win_lose(df_concated)

            if args.latex:
                df_concated = df_concated.map(lambda x: f"{x:.3f}")

                print(df_concated.to_latex())
                print()

            pickle.dump(
                model_to_dfphenomenon,
                open(f"data/tmp/{args.data}_model_to_dfphenomenon.pkl", "wb"),
            )
            if not args.no_figure_update:
                draw_radarcharts(model_to_dfphenomenon, args.data)
        else:
            pickle.load(open(f"data/tmp/{args.data}_model_to_dfphenomenon.pkl", "rb"))

    elif args.x == "paradigm":
        assert args.data == "blimp"

        dfparadigms = []  # dfparadigm: approach x paradigm

        for model in models:
            dfparadigm = accuracy_by(
                args.data, model, approaches, best_templates, "UID"
            )
            dfparadigms.append(dfparadigm)

        # Build a long DataFrame
        df_concated = pd.concat(dfparadigms, axis=0)
        df_concated["method"] = df_concated.index.str.split(", ").map(lambda x: x[0])
        df_concated["model"] = df_concated.index.str.split(", ").map(lambda x: x[1])
        df_long = df_concated.melt(
            id_vars=["model", "method"], var_name="paradigm", value_name="accuracy"
        )

        # Build a DataFrame by paradigm
        lendiff_file = f"results/{args.data}/lendiff_by_paradigm.csv"
        df_accdiff_lendiff = get_df_accdiff_lendiff(df_long, lendiff_file)
        out_accdiff_file = f"results/{args.data}/accdiff_by_paradigm.csv"
        df_accdiff_lendiff.round(3).to_csv(out_accdiff_file, index=False)

        # Add a column for bowmatchratio
        paradigm_to_bowmatchratio = pd.read_csv(
            "data/blimp/paradigm_to_bowmatchratio.csv"
        )
        df_accdiff_lendiff = df_accdiff_lendiff.merge(
            paradigm_to_bowmatchratio, on="paradigm", how="left"
        )
        df_accdiff_lendiff["bowmatches"] = df_accdiff_lendiff["bowmatchratio"] == 1

        # Correlation between accdiff and other columns
        corr_by_model = get_corr_by_model(models, df_accdiff_lendiff)
        # pprint(corr_by_model)

        if not args.no_figure_update:
            # Draw scatter plots to see correlation between length difference and accuracy difference
            out_pdf = f"results/{args.data}/accdiff_by_lendiff_and_paradigm.pdf"
            scatterplot_lendiff_and_accdiff(df_accdiff_lendiff, corr_by_model, out_pdf)

        # Average over models
        df_accmean = df_accdiff_lendiff.groupby("paradigm").agg(
            {
                "paradigm": "first",
                "phenomenon": "first",
                "accdiff": "mean",
                "ll_intemplate_single": "mean",
                "prompting_yn": "mean",
                "lendiff": "mean",
                "bowmatchratio": "mean",
            }
        )
        df_accmean.sort_values("accdiff", ascending=False, inplace=True)
        accdiff_mean_file = Path(out_accdiff_file).with_suffix(".mean.csv")
        df_accmean.round(3).to_csv(accdiff_mean_file, index=False)


def accuracy_by(
    data_name: str,
    model: str,
    approaches: list[str],
    best_templates: pd.DataFrame,
    by: str = "phenomenon",
) -> pd.DataFrame:
    res = {}
    for approach in approaches:
        srs = _compute_accuracy_of_approach_by(
            data_name, model, approach, best_templates, by=by
        )
        res[get_rowname(approach, model)] = srs

    res = pd.DataFrame(res).T

    if by == "phenomenon":
        human_accuracies = json.load(open(f"data/{data_name}/human_accuracies.json"))
        res.loc["Human"] = human_accuracies

    return res


def _compute_accuracy_of_approach_by(
    data_name: str, model: str, approach: str, best_templates: pd.DataFrame, by: str
) -> pd.Series:
    if approach not in ["ll", "ll_pen"]:
        # Append template info after the approach name
        model_abbrev = model_to_abbrev[model.split("_")[0]]
        approach = best_templates.loc[approach, model_abbrev]

    in_dir = f"results/{data_name}"
    result_dir = get_result_dir(in_dir, approach, model)
    result = load_result_latest(result_dir)

    outputs = add_iscorrect(result["outputs"])
    output_df = pd.DataFrame(outputs)

    if by == "phenomenon":
        if data_name == "blimp":
            output_df["phenomenon"] = output_df["UID"].map(uid_to_phenomenon)
        # CLiMP has the `phenomenon` column in the outputs

    res = output_df.groupby(by)["is_correct"].mean()

    return res


def get_rowname(approach: str, model: str) -> str:
    modelname = model_to_abbrev[model.split("_")[0]]
    if "__" in approach:
        approach_base, template = approach.split("__")
        template_i = template.split("_")[-1]
        approachname = approachname_to_officialname[approach_base]
        rowname = f"{approachname}, template {template_i}, {modelname}"
    else:
        approachname = approachname_to_officialname[approach]
        rowname = f"{approach}, {modelname}"

    return rowname


# def interapproach_agreement(in_dir: str, approach_model_pairs: list) -> pd.DataFrame:
#     mat = np.ones((len(approach_model_pairs), len(approach_model_pairs)))

#     for i, (approach_1, model_1) in enumerate(approach_model_pairs):
#         result_dir_1 = get_result_dir(in_dir, approach_1, model_1)
#         outputs_1 = load_result_latest(result_dir_1)["outputs"]

#         for j, (approach_2, model_2) in enumerate(approach_model_pairs):
#             if i == j:
#                 continue

#             result_dir_2 = get_result_dir(in_dir, approach_2, model_2)
#             outputs_2 = load_result_latest(result_dir_2)["outputs"]

#             mat[i, j] = compute_agreement(outputs_1, outputs_2)

#     row_col_names = [
#         get_rowname(approach, model) for approach, model in approach_model_pairs
#     ]
#     res = pd.DataFrame(mat, index=row_col_names, columns=row_col_names).map(
#         lambda x: f"{x:.3f}"
#     )

#     return res


# def compute_agreement(outputs_1: list[dict], outputs_2: list[dict]) -> float:
#     """
#     Compute the accuracy as the agreement between two outputs.
#     """
#     flags = []
#     for output_1, output_2 in zip(outputs_1, outputs_2):
#         # The format of "pred" is different between approaches, so we compare "pred" and "gold" in each approach
#         is_correct_1 = output_1["pred"] == (
#             output_1.get("gold") or output_1["sentence_good"]
#         )
#         is_correct_2 = output_2["pred"] == (
#             output_2.get("gold") or output_2["sentence_good"]
#         )

#         # This condition means that both predictions are correct or both are incorrect
#         if is_correct_1 == is_correct_2:
#             flags.append(1)
#         else:
#             flags.append(0)

#     accuracy = np.mean(flags)

#     return accuracy


def concat_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    df_concated = pd.concat(dfs, axis=0)
    df_concated = df_concated.drop_duplicates(keep="last")
    df_concated["index_1"] = df_concated.index.str.split(", ").map(lambda x: x[-1])
    df_concated["index_2"] = df_concated.index.str.split(", ").map(
        lambda x: ", ".join(x[:-1])
    )
    df_concated.set_index(["index_1", "index_2"], inplace=True)

    return df_concated


def count_win_lose(df: pd.DataFrame) -> None:
    df.query("index_1 != 'Human'", inplace=True)
    df.reset_index(["index_1", "index_2"], inplace=True)
    df_long = df.melt(
        id_vars=["index_1", "index_2"],
        value_vars=df.columns,
        var_name="phenomenon",
        value_name="accuracy",
    )
    df_long.set_index(["index_1", "index_2", "phenomenon"], inplace=True)
    df_long["accuracy"] = df_long["accuracy"].astype(float)
    print(df_long)

    df_lp = df_long.query("index_2 == 'll_intemplate_single'").droplevel("index_2")
    df_pr = df_long.query("index_2 == 'prompting_yn'").droplevel("index_2")
    df_lp["lp_wins"] = df_lp["accuracy"] - 0.01 > df_pr["accuracy"]
    df_lp["pr_wins"] = df_pr["accuracy"] - 0.01 > df_lp["accuracy"]
    print(df_lp)

    table = df_lp.pivot_table(
        index=["phenomenon"], values=["lp_wins", "pr_wins"], aggfunc="sum"
    ).sort_values("lp_wins")
    print(table)


def draw_radarcharts(
    model_to_dfphenomenon: dict[str, pd.DataFrame],
    data_name: str = "blimp",
    layout: str = "3x4",
) -> None:
    if layout == "3x4":
        n_rows = 3
        n_cols = 2
        height = 630
        width = 450
        # Args for make_subplots
        horizontal_spacing = 0.22
        vertical_spacing = 0.14
    elif layout == "2x6":
        n_rows = 2
        n_cols = 3
        height = 420
        width = 680
        # Args for make_subplots
        horizontal_spacing = 0.14
        vertical_spacing = 0.18
    # The plan is to put the chart of blimp on the lower-left part of the climp chart
    if data_name == "blimp":
        # Make the legend at the top small so that it fits in a single row
        legend_font_size = 12
    elif data_name == "climp":
        n_cols *= 2
        width *= 2
        horizontal_spacing /= 2
        legend_font_size = 13
        # Spacer for the BLiMP plots
        model_to_dfphenomenon_list = list(model_to_dfphenomenon.items())
        model_to_dfphenomenon_dummy = [
            (str(i), model_to_df)
            for i, model_to_df in enumerate(model_to_dfphenomenon.values())
        ]
        if n_cols == 6:
            model_to_dfphenomenon = dict(
                model_to_dfphenomenon_dummy[:3]
                + model_to_dfphenomenon_list[:3]
                + model_to_dfphenomenon_dummy[3:]
                + model_to_dfphenomenon_list[3:]
            )
        elif n_cols == 4:
            model_to_dfphenomenon = dict(
                model_to_dfphenomenon_dummy[:2]
                + model_to_dfphenomenon_list[:2]
                + model_to_dfphenomenon_dummy[2:4]
                + model_to_dfphenomenon_list[2:4]
                + model_to_dfphenomenon_dummy[4:6]
                + model_to_dfphenomenon_list[4:6]
            )

    fig = make_subplots(
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(model_to_dfphenomenon.keys()),
        specs=[[{"type": "polar"}] * n_cols] * n_rows,
    )
    fig.update_layout(
        # This is the width of the plot or container in centi-inches. The original width is 700
        # https://plotly.com/python/reference/layout/
        width=width,
        height=height,
        # Set the left and right margings to make the plots smaller
        margin=dict(t=0, b=24, l=48, r=50),  # Changing t doesn't affect the plot
        # xanchor="center" means that the legend's center is at `x`
        # By default xref="paper", meaning that `x` refers to the plotting area
        # So, x is 0.5 with reference to "paper" in this case
        # yanchor="top" means that the legend's top is at `y` (0 with reference to "paper")
        legend=dict(
            xanchor="center",
            x=0.5,  # + 0.19,  # [ARR submission] Move x from 0.5 so that the legend is at the center of the figure as a whole
            yanchor="bottom",
            y=1.1,
            orientation="h",
            # Make the font larger
            font=dict(size=legend_font_size),
        ),
    )
    # Make subplot titles smaller
    fig.update_annotations(font_size=13)
    # Adjust position for each subplot title
    for anno in fig.layout.annotations:
        anno.update(y=anno.y + 0.04)

    # Used for ARR submisison
    # elif data_name == "climp":
    #     n_rows = 2
    #     n_cols = 1
    #     fig = make_subplots(
    #         rows=n_rows,
    #         cols=n_cols,
    #         horizontal_spacing=0.14,
    #         vertical_spacing=0.1,  #
    #         subplot_titles=list(model_to_dfphenomenon.keys()),
    #         specs=[[{"type": "polar"}] * n_cols] * n_rows,
    #     )
    #     fig.update_layout(
    #         width=220,
    #         height=420,
    #         margin=dict(t=46, b=24, l=48, r=46),
    #         showlegend=False,
    #         # legend=dict(
    #         #     xanchor="center",
    #         #     x=0.5,
    #         #     yanchor="top",
    #         #     y=-0.12,
    #         #     orientation="h",
    #         #     font=dict(size=13),
    #         # ),
    #     )
    #     # Make subplot titles smaller
    #     fig.update_annotations(font_size=13)

    #     # Adjust position for each subplot title
    #     for _, annotation in enumerate(fig.layout.annotations):
    #         annotation.update(y=annotation.y + 0.01)  #

    dct = {
        "ll_intemplate_single": "In-template LP",
        "prompting_yn": "Yes/No prob comp",
        "Human": "Humans",
    }

    for df_i, df in enumerate(model_to_dfphenomenon.values(), 0):
        phenomena = list(df.columns)

        df.index = df.index.str.split(", ").map(lambda x: x[0])

        if df_i == 0:
            showlegend = True
        else:
            showlegend = False

        for row_i in range(len(df)):
            values = list(df.iloc[row_i])

            # Use the same color for the same approach across radar charts
            if df.index[row_i] == "ll_pen":
                line = dict(color="#19C585")
            elif df.index[row_i] == "ll_intemplate_single":
                line = dict(color="#5051f9")
            elif df.index[row_i] == "prompting_yn":
                line = dict(color="#e93d2e")  # Red
            elif df.index[row_i] == "Human":
                line = dict(color="rgba(0, 0, 0, 0.2)")
            else:
                raise NotImplementedError(f"Unknown approach: {df.index[row_i]}")
                # Automatically determine the line color
                # line = None

            line.update(width=1.4)

            fig.add_trace(
                go.Scatterpolar(
                    # The first element should be repeated to close the polygon
                    r=values + [values[0]],
                    theta=phenomena + [phenomena[0]],
                    fill="none",
                    name=dct[df.index[row_i]],
                    marker=dict(size=4),
                    line=line,
                    showlegend=showlegend,
                ),
                row=df_i // n_cols + 1,
                col=df_i % n_cols + 1,
            )

    fig.update_polars(
        radialaxis=dict(
            tickangle=0,
            # Make ticks smaller
            # tickfont=dict(size=12),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0", "", "0.5", "", "1.0"],
        )
    )

    # Keep the message "Loading [MathJax]/extensions" from appearing
    # See https://github.com/plotly/plotly.py/issues/3469#issuecomment-1081736804
    pio.full_figure_for_development(fig, warn=False)
    # Save the figure in pdf
    pdf_file = f"results/{data_name}/accuracy_by_phenomenon_{data_name}.pdf"
    fig.write_image(pdf_file)


def get_df_accdiff_lendiff(df_long: pd.DataFrame, lendiff_file: str) -> pd.DataFrame:
    df = df_long.pivot(index=["model", "paradigm"], columns="method", values="accuracy")
    phenom_to_abbrev = data_to_phenomenon_to_abbrev["blimp"]
    df["phenomenon"] = (
        df.index.get_level_values("paradigm")
        .map(uid_to_phenomenon)
        .map(phenom_to_abbrev)
    )
    df["accdiff"] = df["prompting_yn"] - df["ll_intemplate_single"]
    df = df[["phenomenon", "accdiff", "ll_intemplate_single", "prompting_yn"]]

    # Merge lendiff
    df_lendiff = pd.read_csv(lendiff_file)
    df_merged = df.merge(df_lendiff, on=["model", "paradigm"], how="left")
    df_merged.sort_values(["model", "accdiff"], ascending=False, inplace=True)

    return df_merged


def get_corr_by_model(
    models: list[str], df_accdiff_lendiff: pd.DataFrame
) -> pd.DataFrame:
    res = []
    for model in models:
        model_abbrev = model_to_abbrev[model.split("_")[0]]
        df_accdiff_lendiff_model = df_accdiff_lendiff.query(
            f"model == '{model_abbrev}'"
        )
        table = df_accdiff_lendiff_model[["accdiff", "lendiff", "bowmatchratio"]].corr()
        res.append(
            {
                "model": model_abbrev,
                "accdiff-lendiff": table.iloc[0, 1].round(3),
                "accdiff-bowmatchratio": table.iloc[0, 2].round(3),
            }
        )

    return pd.DataFrame(res)


def scatterplot_lendiff_and_accdiff(
    df_accdiff_lendiff: pd.DataFrame,
    corr_by_model: pd.DataFrame,
    out_pdf: str,
) -> None:
    # Draw scatter plots
    df_accdiff_lendiff["facet_order"] = df_accdiff_lendiff["model"].map(
        {
            "Llama-3": 1,
            "Mixtral": 2,
            "Qwen2": 3,
            "Llama-3-Instruct": 4,
            "Mixtral-Instruct": 5,
            "Qwen2-Chat": 6,
        }
    )
    df_accdiff_lendiff.sort_values("facet_order", inplace=True)
    # df_merged = df_merged.query("lendiff != 0")

    g = sns.FacetGrid(
        df_accdiff_lendiff,
        col="model",
        hue="phenomenon",
        # Setting hue_order and hue_kws for sns.FacetGrid does not change markers, though it seems to work according to pages on the Web
        # https://github.com/mwaskom/seaborn/issues/715
        # The seaborn document explains hue_kws as: Other keyword arguments to insert into the plotting call to let other plot attributes vary across levels of the hue variable (e.g. the markers in a scatterplot).
        hue_order=df_accdiff_lendiff["phenomenon"].unique(),
        hue_kws={
            "markers": ["o", "v", "^", "<", ">", "8", "s", "p", "*", "h", "H", "d"]
        },
        col_wrap=3,
    )
    g.map_dataframe(
        sns.scatterplot,
        x="lendiff",
        y="accdiff",
        # The codes below change markers but cause wrong assignments of colors and markers to phenomena
        # hue="phenomenon",
        # style="phenomenon",
        # markers=["o", "v", "^", "<", ">", "8", "s", "p", "*", "h", "H", "d"],
        s=18,
    )
    g.set_titles("{col_name}")
    # Set labels
    g.set_xlabels("Token-length difference")
    g.set_ylabels("Accuracy difference")

    def annotate(data, **kws):
        # Avoid overlapping texts
        if data["phenomenon"].iloc[0] != "Arg. str.":
            return

        ax = plt.gca()
        df = corr_by_model.query(f"model == '{data['model'].iloc[0]}'")
        corr = df["accdiff-lendiff"].iloc[0]
        ax.text(0.72, 0.86, f"r = {corr:.3f}", transform=ax.transAxes)

    g.map_dataframe(annotate)

    g.add_legend()
    sns.move_legend(
        g,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=6,
        title=None,
        frameon=False,
    )

    g.figure.set_size_inches(10, 6.4)
    g.figure.subplots_adjust(top=0.86, left=0.07, right=0.99)

    plt.savefig(out_pdf)


if __name__ == "__main__":
    main()
