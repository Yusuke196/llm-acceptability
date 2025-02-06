import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import get_dataloader, get_dataset
from src.modeling import get_model, get_tokenizer
from src.utils import (
    get_out_file,
    get_start_time,
    load_logprobs_latest,
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
        default="ll",
        choices=[
            "ll",
            "ll_normalized",  # MeanLP in the terminology of Lau+ 2020
            "ll_pen",  # PenLP in the terminology of Lau+ 2020
            "ll_intemplate_single",
            "ll_intemplate_single_normalized",
            "ll_intemplate_single_pen",
            "ll_intemplate_compar",
            "ll_intemplate_compar_normalized",
            "ll_intemplate_compar_pen",
        ],
        help="ll_intemplate_single: Compute log-likelihoods of each sentence in the sentence, ll_intemplate_compar: Compute log-likelihoods of texts that comparatively evaluate the two sentences.",
    )
    parser.add_argument(
        "--template_file",
        "-t",
        type=str,
    )
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
        "--target", type=str, choices=["whole_seq", "sentence"], default="whole_seq"
    )
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
    dataset = get_dataset(
        args.data, args.approach, tokenizer, args.template_file, args.debug
    )

    if args.logprobs_dir is None:
        # Inference without using cached logprobs
        # Prepare for inference
        set_seed(args.seed)  # Note on experiments: This was introduced on 17th April

        if not torch.cuda.is_available():
            print("CUDA is not available. Exiting...")
            return

        device = f"cuda:{args.gpu}"
        model = get_model(args.model, args.quantize_type, tokenizer)

        # Prepare for logging
        start_time = get_start_time(device)

        # Computation
        dataloader = get_dataloader(
            dataset, args.batch_size, tokenizer.pad_token_id, args.data
        )
        result = compute_logprobs(dataloader, tokenizer, model, args.target)

        # Logging
        log_inference_time(start_time)
        log_memory_usage(device)

        # Judgment
        outputs = judge(
            dataset, result["logprobs_good"], result["logprobs_bad"], args.approach
        )
    else:
        # Inference using cached logprobs
        logprobs_dir = (
            Path(args.logprobs_dir) / f'{args.model.split("/")[1]}_{args.quantize_type}'
        )
        if args.approach not in ["ll", "ll_normalized", "ll_pen"]:
            logprobs_dir = logprobs_dir / Path(args.template_file).stem

        logprobs = load_logprobs_latest(logprobs_dir, args.target)

        outputs = judge(
            dataset, logprobs["logprobs_good"], logprobs["logprobs_bad"], args.approach
        )

    # Evaluation
    outputs = clean_outputs(outputs)
    scores = compute_scores(outputs)

    # Save scores, args, and outputs
    out_file = get_out_file(args)
    write_result(scores, args, outputs, out_file)

    # Save logprobs and tokens (labels)
    if args.logprobs_dir is None:
        if args.target == "sentence":
            out_file_part = "logprobs"
        elif args.target == "whole_seq":
            out_file_part = "logprobs_whole"

        out_file_parts = [
            out_file.name.split(".")[0],
            out_file_part,
            ".".join(out_file.name.split(".")[1:]),
        ]
        out_file_2 = out_file.with_name(".".join(out_file_parts))
        with open(out_file_2, "w") as f:
            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False,
            )


def compute_logprobs(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    target: str,
) -> defaultdict:
    """
    Compute log probabilities for each token in each sentence in `dataset`
    """
    res = defaultdict(list)
    for batch in tqdm(dataloader):
        for s in ["good", "bad"]:
            logprobs_batch, tokens_batch, mask = _compute_logprobs_batch(
                batch[f"ids_{s}"], batch[f"mask_{s}"], tokenizer, model, target
            )
            res[f"logprobs_{s}"].extend(logprobs_batch)
            res[f"tokens_{s}"].extend(tokens_batch)
            if target == "whole_seq":
                res[f"mask_{s}"].extend(mask)

    return res


def _compute_logprobs_batch(
    ids: torch.Tensor,
    mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    target: str,
) -> tuple[list[list[float]], list[list[str]], list[list[int]]]:
    """
    Compute log probabilities for each token in each batch
    """
    with torch.no_grad():
        outputs = model(ids, labels=ids)

    # Prepare shifted logits and labels to compute the loss of next token prediction
    # outputs.logits.shape == (batch_size, seq_len, vocab_size)
    shifted_logits = outputs.logits[:, :-1, :].contiguous()
    shifted_labels = ids[:, 1:].contiguous()
    # Use cross-entropy loss function to easily compute the log probs
    # A flattened tensor is returned here
    losses = F.cross_entropy(
        # shifted_logits.size(-1) is the vocab size
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction="none",
    )
    logprobs = -losses.reshape(shifted_labels.shape)

    shifted_mask = mask[:, 1:].contiguous()

    if target == "sentence":
        logprobs = [
            logprobs_sent[mask == 1].tolist()
            for logprobs_sent, mask in zip(logprobs, shifted_mask)
        ]
        tokens = [
            tokenizer.convert_ids_to_tokens(labels_sent[mask == 1])
            for labels_sent, mask in zip(shifted_labels, shifted_mask)
        ]
        mask = None
    elif target == "whole_seq":
        # Include template parts in `logprobs` and `tokens`
        logprobs = [
            logprobs_sent[labels_sent != tokenizer.pad_token_id].tolist()
            for logprobs_sent, labels_sent in zip(logprobs, shifted_labels)
        ]
        tokens = [
            tokenizer.convert_ids_to_tokens(
                labels_sent[labels_sent != tokenizer.pad_token_id]
            )
            for labels_sent in shifted_labels
        ]
        mask = [
            mask_sent[labels_sent != tokenizer.pad_token_id].tolist()
            for mask_sent, labels_sent in zip(shifted_mask, shifted_labels)
        ]

    return logprobs, tokens, mask


def judge(
    dataset: Dataset,
    logprobs_good: list[list[float]],
    logprobs_bad: list[list[float]],
    approach: str,
) -> Dataset:
    good_sentences = dataset["sentence_good"]
    bad_sentences = dataset["sentence_bad"]

    def _judge_for_sent(
        i: int,
    ) -> str:
        logprobs_sent_good = _compute_logprob_for_sent(logprobs_good[i], approach)
        logprobs_sent_bad = _compute_logprob_for_sent(logprobs_bad[i], approach)

        return (
            good_sentences[i]
            if logprobs_sent_good > logprobs_sent_bad
            else bad_sentences[i]
        )

    parallel = Parallel()
    preds = parallel(delayed(_judge_for_sent)(i) for i in range(len(logprobs_good)))

    dataset = dataset.add_column("pred", preds)

    return dataset


def _compute_logprob_for_sent(logprobs: list[float], approach: str) -> float:
    if approach.endswith("_normalized"):
        # MeanLP in the terminology of Lau+ 2020
        return np.mean(logprobs)
    elif approach.endswith("_pen"):
        # PenLP in the terminology of Lau+ 2020
        alpha = 0.8
        denom = ((5 + len(logprobs)) / (5 + 1)) ** alpha
        return np.sum(logprobs) / denom
    else:
        return np.sum(logprobs)


def clean_outputs(outputs: Dataset) -> list[dict]:
    outputs = outputs.remove_columns(
        ["ids_good", "mask_good", "ids_bad", "mask_bad"]
    ).to_list()

    return outputs


def compute_scores(outputs: list[dict]) -> dict:
    accuracy = np.mean(
        [1 if output["pred"] == output["sentence_good"] else 0 for output in outputs]
    ).round(6)

    return {"accuracy": accuracy}


if __name__ == "__main__":
    main()
