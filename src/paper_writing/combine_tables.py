import argparse
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--in_files", "-i", type=str, nargs="+", required=True)
    parser.add_argument("--blimp_file", "-b", type=str, required=True)
    parser.add_argument("--climp_file", "-c", type=str, required=True)
    parser.add_argument(
        "--drop_columns", "-d", type=str, nargs="+", default=[]
    )  # "Mixtral" "Mixtral-Instruct" "Yi-1.5" "Yi-1.5-Chat"
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    # dfs = [pd.read_csv(file, index_col=0) for file in args.in_files]
    blimp = pd.read_csv(args.blimp_file, index_col=0)
    climp = pd.read_csv(args.climp_file, index_col=0)
    climp.columns = [
        f"{col}_climp" if col in blimp.columns else col for col in climp.columns
    ]
    merged = pd.merge(blimp, climp, left_index=True, right_index=True)
    merged.drop(columns=args.drop_columns, inplace=True)

    s = merged.to_latex()
    s = "    " + s.replace("\n", "\n    ")
    print(s)


if __name__ == "__main__":
    main()
