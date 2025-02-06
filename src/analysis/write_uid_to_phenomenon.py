import json
from pathlib import Path
from utils import read_jsonl

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/blimp/data")
    parser.add_argument(
        "--out_file", type=str, default="data/blimp/uid_to_phenomenon.json"
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    res = {}

    for in_file in Path(args.in_dir).glob("*.jsonl"):
        obj = read_jsonl(in_file)[0]
        phenomenon = obj["linguistics_term"]
        res[in_file.stem] = phenomenon

    with open(args.out_file, "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
