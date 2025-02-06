import argparse
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import get_dataset_with_prompts
from src.modeling import get_model, get_tokenizer
from src.utils import (
    get_out_file,
    get_start_time,
    log_inference_time,
    log_memory_usage,
    set_seed,
    write_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument("--data", "-d", type=str, default="blimp")
    parser.add_argument(
        "--approach",
        "-a",
        type=str,
        default="prompting_yn",
    )
    parser.add_argument(
        "--template_file",
        "-t",
        type=str,
    )
    parser.add_argument("--examples_file", "-e", type=str)
    # Model and inference
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
    )
    parser.add_argument(
        "--quantize_type",
        "-q",
        type=str,
        default="none",
        choices=["4bit", "8bit", "none"],
    )
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default="results",
        help="A result json file and logprobs json file are saved under this directory.",
    )
    # Others
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--logprobs_dir",
        type=str,
        help="If specified, load logprobs and other data from a directory under this directory to perform judgment.",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    print("args:")
    pprint(vars(args))

    tokenizer = get_tokenizer(args.model)
    dataset = get_dataset_with_prompts(
        args.data,
        args.template_file,
        args.examples_file,
        tokenizer,
        args.model,
        args.debug,
    )

    # Prepare for inference
    set_seed(args.seed)

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return

    device = f"cuda:{args.gpu}"
    model = get_model(args.model, args.quantize_type, tokenizer)

    # Prepare for logging
    start_time = get_start_time(device)

    # Computation
    outputs = compute_yesprobs(
        model, dataset, args.batch_size, tokenizer, args.template_file
    )

    # Logging
    log_inference_time(start_time)
    log_memory_usage(device)

    scores = compute_scores(outputs)

    # Save scores, args, and outputs
    out_file = get_out_file(args)
    write_result(scores, args, outputs, out_file)


def compute_yesprobs(
    model: AutoModelForCausalLM,
    dataset: Dataset,
    batch_size: int,
    tokenizer: AutoTokenizer,
    template_file: str,
) -> Dataset:
    dataloader = DataLoader(dataset, batch_size=batch_size)
    yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
    no_token_id = tokenizer.convert_tokens_to_ids("No")
    assert yes_token_id is not None
    assert no_token_id is not None

    # elif "climp" in template_file:
    #     # convert_tokens_to_ids("是") returns None, so use encode() instead
    #     yes_encoded = tokenizer.encode("是")
    #     no_encoded = tokenizer.encode("否")
    #     assert len(yes_encoded) == 1
    #     assert len(no_encoded) == 1
    #     yes_token_id = yes_encoded[0]
    #     no_token_id = no_encoded[0]

    goodorbad_to_probslist = defaultdict(list)
    for batch in tqdm(dataloader):
        for good_or_bad in ["good", "bad"]:
            inputs = tokenizer.batch_encode_plus(
                batch[f"prompt_{good_or_bad}"], return_tensors="pt", padding=True
            )
            with torch.no_grad():
                logits = model(**inputs).logits
                next_token_logits = logits[:, -1, :]
                yes_logits = next_token_logits[:, yes_token_id]
                no_logits = next_token_logits[:, no_token_id]
                yes_probs = yes_logits / (yes_logits + no_logits)
                goodorbad_to_probslist[good_or_bad] += yes_probs.tolist()

    outputs = (
        dataset.add_column("yesprob_good", goodorbad_to_probslist["good"])
        .add_column("yesprob_bad", goodorbad_to_probslist["bad"])
        .to_list()
    )

    return outputs


def compute_scores(outputs: list[dict]) -> dict:
    accuracy = np.mean(
        [
            1 if output["yesprob_good"] > output["yesprob_bad"] else 0
            for output in outputs
        ]
    ).round(6)

    return {"accuracy": accuracy}


if __name__ == "__main__":
    main()
