import argparse

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
        "--out_pdf", "-o", type=str, default="results/blimp/accuracy_by_attractor.pdf"
    )
    args = parser.parse_args()

    return args


attractortype_to_paradigms = {
    "No attractor": [
        "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2",
    ],
    "Relational noun": ["distractor_agreement_relational_noun"],
    "Relative clause": ["distractor_agreement_relative_clause"],
}


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.in_file)

    def get_attractortype(paradigm) -> str:
        for attractortype, paradigms in attractortype_to_paradigms.items():
            if paradigm in paradigms:
                return attractortype

        return "NA"

    df["attractor_type"] = df["paradigm"].apply(get_attractortype)
    df_2 = (
        df.pivot_table(
            index=["model", "attractor_type"],
            values=["ll_intemplate_single", "prompting_yn"],
            aggfunc="mean",
        )
        .drop("NA", level="attractor_type")
        .reset_index()
        .melt(
            id_vars=["model", "attractor_type"],
            value_vars=["ll_intemplate_single", "prompting_yn"],
            var_name="method",
            value_name="acc",
        )
    )
    print(df_2.round(3))

    draw_barplot(df_2, args.out_pdf)

    df_mean = df_2.pivot_table(
        index="attractor_type",
        aggfunc="mean",
        values="acc",
    )
    print("Mean by attractor type:")
    print(df_mean.round(3))


def draw_barplot(df: pd.DataFrame, out_pdf: str) -> None:
    # Sort by models and attractor types
    df["model_order"] = df["model"].map(
        {
            "Llama-3": 1,
            "Mixtral": 2,
            "Qwen2": 3,
            "Llama-3-Instruct": 4,
            "Mixtral-Instruct": 5,
            "Qwen2-Chat": 6,
        }
    )
    df["attrtype_order"] = df["attractor_type"].map(
        {
            "No attractor": 1,
            "Relational noun": 2,
            # "Prep. p.": 2,
            "Relative clause": 3,
        }
    )
    df.sort_values(
        ["attrtype_order", "model_order"],
        inplace=True,
    )
    # Rename methods
    df["method"] = df["method"].map(
        {
            "ll_intemplate_single": "I. LP",
            "prompting_yn": "Y/N p.c.",
        }
    )

    g = sns.FacetGrid(df, col="model")
    g.map_dataframe(sns.barplot, x="method", y="acc", hue="attractor_type")
    g.set_titles("{col_name}")
    g.set(xlabel="", ylabel="Accuracy")  # , ylim=(0.5, 1)
    sns.set_palette("colorblind")

    g.add_legend()
    sns.move_legend(
        g,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        title=None,
        frameon=False,
    )
    g.figure.set_size_inches(9.8, 2)
    g.figure.subplots_adjust(top=0.75, bottom=0.13, left=0.06, right=0.99)

    plt.savefig(out_pdf)


if __name__ == "__main__":
    main()
