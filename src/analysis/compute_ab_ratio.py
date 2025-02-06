import argparse

import pandas as pd
from joblib import Parallel, delayed

from src.constants import (
    data_to_approaches,
    data_to_models,
    model_to_abbrev,
)
from src.analysis.print_scores import summarize_results
from src.utils import get_result_dir, load_result_latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="blimp")
    parser.add_argument("--in_file", "-i", type=str)
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    in_dir = f"results/{args.data}"
    approaches = [a for a in data_to_approaches[args.data] if "ab" in a]
    models = data_to_models[args.data]

    df_all = get_df_all(
        in_dir,
        approaches,
        models,
        abbreviate_col_names=True,
    )
    print(df_all)
    print()

    df_summary = summarize_results(
        df_all, main_stat="mean", show_percent=True, sort_rows=False
    )
    print(df_summary)
    print()

    s = df_summary.to_latex(float_format="{:.3f}".format)
    s = "    " + s.replace("\n", "\n    ")
    print(s)
    print()


def get_df_all(
    in_dir: str,
    approaches: list[str],
    models: list[str],
    abbreviate_col_names: bool,
) -> pd.DataFrame:
    def get_row(approach: str) -> dict:
        row = {}
        for model in models:
            result_dir = get_result_dir(in_dir, approach, model)
            result = load_result_latest(result_dir)
            model_s = model.split("_")[0]  # Remove quantization info

            if result is not None:
                preds = [output["pred"] for output in result["outputs"]]
                row[model_s] = sum(1 for pred in preds if pred == "A") / len(preds)
            else:
                row[model_s] = None

        return row

    parallel = Parallel(-1)
    rows = parallel(delayed(get_row)(approach) for approach in approaches)
    df = pd.DataFrame(rows, index=approaches)

    if abbreviate_col_names:
        df.columns = [model_to_abbrev[model] for model in df.columns]

    return df


if __name__ == "__main__":
    main()
