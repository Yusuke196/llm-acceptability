import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="blimp")
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    intemp_prefixes = [
        ("templates_ll", f"template_{args.data}_single"),
        ("templates_ll", f"template_{args.data}_compar"),
    ]
    prompting_prefixes = [
        ("prompts", f"template_{args.data}_ab"),
        ("prompts", f"template_{args.data}_yn"),
    ]

    records = []
    for in_dir, in_file_prefix in intemp_prefixes:
        for in_file in Path(in_dir).rglob(f"{in_file_prefix}*"):
            records.append(
                {
                    "in_file": in_file.name,
                    "text": in_file.read_text(),
                }
            )

    for in_dir, in_file_prefix in prompting_prefixes:
        for in_file in Path(in_dir).rglob(f"{in_file_prefix}*"):
            records.append(
                {
                    "in_file": in_file.name,
                    "text": json.load(in_file.open())["user_template"],
                }
            )

    df = pd.DataFrame(records).sort_values("in_file")

    test_prompt_templates(df, args.data)

    # Print all the templates
    # Remove templates specifically for base models as they overlap with the ones for instruct models
    df.query(
        'not in_file.str.contains("ab_") and not in_file.str.contains("yn_")',
        inplace=True,
    )
    print(df)


def test_prompt_templates(df: pd.DataFrame, dataname: str) -> None:
    for typ in ["ab", "yn"]:
        for i in range(1, 6):
            base = df.query(f'in_file == "template_{dataname}_{typ}_{i}.json"')[
                "text"
            ].values[0]
            inst = df.query(f'in_file == "template_{dataname}_{typ}-chat_{i}.json"')[
                "text"
            ].values[0]

            if dataname == "blimp":
                assert base == inst + "\nAnswer: ", ("type:", typ, "i:", i)
            elif dataname == "climp":
                assert base == inst + "\n答案：", ("type:", typ, "i:", i)


if __name__ == "__main__":
    main()
