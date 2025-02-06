import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def read_jsonl(file_name: str) -> list:
    with jsonlines.open(file_name) as reader:
        return [line for line in reader]


def replace_words(text: str, dic: dict) -> str:
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


# For inference


def check_args(args: argparse.Namespace) -> None:
    if "Swallow" in args.model and "_en" in args.template_file:
        print("Warning: A Swallow model may not work well with an English template.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# For logging


def get_start_time(device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)

    return time.perf_counter()


def log_inference_time(start_time: float):
    end_time = time.perf_counter()
    print("\n" + "-" * 30)
    print("Inference Time")
    print(f"{end_time - start_time:.2f} seconds")
    print("-" * 30)


def log_memory_usage(device):
    if device.startswith("cuda"):
        print("\n" + "-" * 30)
        print("Memory Usage")
        max_memory_allocated = torch.cuda.max_memory_allocated(device=device)
        print(f"- Max Memory Allocated: {format_memory(max_memory_allocated)}")
        max_memory_reserved = torch.cuda.max_memory_reserved(device=device)
        print(f"- Max Memory Reserved: {format_memory(max_memory_reserved)}")
        print("-" * 30 + "\n")


def format_memory(bytes):
    gb = bytes / (1024**3)
    return f"{gb:.3f} GB"


# For evaluation


def get_labels(template_file: str) -> list[str]:
    stem = Path(template_file).stem
    if stem.endswith("-en") or "_yn" in stem:
        return ["Yes", "No"]
    elif stem.endswith("-ja"):
        return ["はい", "いいえ"]
    elif "_ab" in stem:
        return ["A", "B"]
    else:
        raise ValueError(f"Unknown template_file: {template_file}")


def clean_pred(outputs: list, labels: list[str]) -> list:
    pattern = r"({})".format("|".join(labels))

    res = []
    for output in outputs:
        if match := re.match(pattern, output["pred"]):
            output["pred"] = match.group(0)

        if labels == ["Yes", "No"]:
            if re.match(r"yes", output["pred"]):
                output["pred"] = "Yes"
            if re.match(r"no", output["pred"]):
                output["pred"] = "No"

        res.append(output)

    return res


def compute_scores(outputs: list[dict], labels: list[str]) -> dict[str, Any]:
    scores = {}

    scores["accuracy"] = np.mean(
        [1 if output["pred"] == output["gold"] else 0 for output in outputs]
    ).round(6)

    wellformatted_outputs = [output for output in outputs if output["pred"] in labels]
    scores["_well-formatted_ratio"] = round(
        len(wellformatted_outputs) / len(outputs), 6
    )

    if wellformatted_outputs != []:
        y_gold = [
            1 if output["gold"] == labels[0] else -1 for output in wellformatted_outputs
        ]
        y_pred = [
            1 if output["pred"] == labels[0] else -1 for output in wellformatted_outputs
        ]
        # Calculate MCC
        scores["mcc"] = round(matthews_corrcoef(y_gold, y_pred), 6)
        # Get confusion matrix
        conf_mat = confusion_matrix(y_gold, y_pred).tolist()
        scores["true_negative_rate"] = round(conf_mat[0][0] / sum(conf_mat[0]), 6)
        scores["true_positive_rate"] = round(conf_mat[1][1] / sum(conf_mat[1]), 6)
        # Join the integers in the inner lists to make it easier to read in JSON
        # 1st row (Gold is No):  True No, False Yes
        # 2nd row (Gold is Yes): False No, True Yes
        scores["_confusion_matrix"] = [
            " ".join([str(i) for i in lst]) for lst in conf_mat
        ]

    scores["notes"] = [
        "The true positive rate is synonymous with recall.",
        "MCC and the confusion matrix are calculated only for the well-formatted outputs.",
    ]

    return scores


def get_out_file(args: argparse.Namespace) -> Path:
    if hasattr(args, "approach"):
        approach = args.approach
    elif not hasattr(args, "examples_file") or args.examples_file is None:
        approach = "0-shot"
    else:
        approach = re.search(r"\d+-shot", args.examples_file).group()

    # When using a model from HuggingFace
    if "/" in args.model:
        model_name = f'{args.model.split("/")[1]}_{args.quantize_type}'
    else:
        model_name = args.model

    if args.debug:
        suffix = ".debug"
    else:
        suffix = ""

    out_dir = Path(args.out_dir) / args.data / approach / model_name
    if args.approach not in ["ll", "ll_normalized", "ll_pen"]:
        out_dir = out_dir / Path(args.template_file).stem

    out_file = out_dir / f'{time.strftime("%Y%m%d-%H%M%S")}{suffix}.json'

    return out_file


def write_result(
    scores: dict[str, Any],
    args: argparse.Namespace,
    outputs: list[dict],
    out_file: Path,
) -> None:
    # Set parents=True to create the parent directories if they don't exist
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        print(f"Writing results to {out_file}")
        args_dict = {
            k: v for k, v in vars(args).items() if k not in ["seed", "out_dir", "debug"]
        }
        result = {
            "scores": scores,
            "args": args_dict,
            "outputs": outputs,
        }
        json.dump(result, f, indent=4, ensure_ascii=False)


def load_logprobs_latest(dir: Path, target: str) -> dict | None:
    if target == "sentence":
        stem_end = "logprobs"
    elif target == "whole_seq":
        stem_end = "logprobs_whole"

    files = [
        file
        for file in sorted(dir.glob("*"))
        if file.stem.endswith(stem_end)  # difference from load_result_latest()
    ]

    if files != []:
        print(f"Loading {files[-1]}")
        return json.load(open(files[-1]))
    else:
        print(f"No logprobs file found in {dir}")
        return None


# For scoring


def get_result_dir(in_dir: str, approach: str, model: str) -> Path:
    if "__" not in approach:
        result_dir = Path(in_dir) / approach / model
    else:
        approach_base, template = approach.split("__")
        result_dir = Path(in_dir) / approach_base / model / template

    return result_dir


def load_result_latest(result_dir: Path) -> dict | None:
    objs = [
        json.load(open(file))
        for file in sorted(result_dir.glob("*"))
        if not re.search(r"(debug|logprobs|logprobs_whole)$", file.stem)
    ]
    if objs == []:
        print(f"No result file found in {result_dir}")
        return None

    return objs[-1]


def add_iscorrect(outputs: list[dict]) -> list[dict]:
    for output in outputs:
        if "yesprob_good" in output:
            output["is_correct"] = output["yesprob_good"] > output["yesprob_bad"]
        elif "pred" in output:
            gold = output.get("gold") or output["sentence_good"]
            output["is_correct"] = output["pred"] == gold
        else:
            raise ValueError("No key to determine `is_correct`.")

    return outputs


def get_uid_to_phenomenon() -> dict:
    uid_to_phenomenon = json.load(open("data/blimp/uid_to_phenomenon.json"))
    # Replace 's-selection' with 'argument_structure'
    uid_to_phenomenon["animate_subject_trans"] = "argument_structure"
    uid_to_phenomenon["animate_subject_passive"] = "argument_structure"

    return uid_to_phenomenon


def format_df(
    df: pd.DataFrame,
    show_percent: bool,
    mark_highest: bool = False,
    mark_second: bool = False,
) -> pd.DataFrame:
    """
    - Mark the highest and second highest score if specified.
    - Format all the scores.
    """
    # Record the index (approach) of the highest and second highest scores
    if mark_highest:
        indices_highest = df.idxmax(axis=0)
    if mark_second:
        indices_second_highest = df.apply(lambda x: x.sort_values().index[-2])

    if show_percent:
        df = df.map(lambda x: f"{x * 100:.1f}")
    else:
        df = df.map(lambda x: f"{x:.3f}".replace("0.", "."))

    if mark_highest:
        for col, idx in indices_highest.items():
            # df.loc[idx, col] = f"$\mathbf{{{df.loc[idx, col]}}}$"  # Not very readable code
            df.loc[idx, col] = "$\mathbf{" + df.loc[idx, col] + "}$"
    if mark_second:
        for col, idx in indices_second_highest.items():
            df.loc[idx, col] = "\\underline{" + df.loc[idx, col] + "}"

    return df


def judge_correctness(record: dict) -> bool:
    if "yesprob_good" in record:
        return record["yesprob_good"] > record["yesprob_bad"]
    else:
        gold = record.get("gold") or record["sentence_good"]
        return record["pred"] == gold
