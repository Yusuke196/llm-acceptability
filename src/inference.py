import argparse
from pprint import pprint

import torch
from datasets import Dataset
from outlines import generate
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from src.dataset import get_dataset_with_prompts
from src.modeling import get_generator, get_pipe, get_tokenizer
from src.utils import (
    check_args,
    clean_pred,
    compute_scores,
    get_labels,
    get_out_file,
    get_start_time,
    log_inference_time,
    log_memory_usage,
    set_seed,
    write_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str)
    parser.add_argument("--approach", "-a", type=str, default="prompting")
    parser.add_argument(
        "--template_file",
        "-t",
        type=str,
    )
    parser.add_argument("--examples_file", "-e", type=str, default=None)
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
    # Note: The max length of BLiMP sentences according to the Llama 2 tokenizer is 63
    # parser.add_argument("--max_len", type=int, default=128)
    # parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--max_new_tokens", type=int, default=1
    )  # Set this to 3 or 4 for Japanese because いいえ can take that many tokens
    parser.add_argument("--use_outlines", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--out_dir", "-o", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    check_args(args)
    # print(f"Running {args.model}...\n")
    print("args:")
    pprint(vars(args))

    # Prepare for inference
    set_seed(args.seed)
    tokenizer = get_tokenizer(args.model)
    dataset = get_dataset_with_prompts(
        args.data,
        args.template_file,
        args.examples_file,
        tokenizer,
        args.model,
        args.debug,
    )

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return

    device = "cuda:0"

    if args.use_outlines:
        generator = get_generator(args.model, args.quantize_type, tokenizer, device)
        start_time = get_start_time(device)

        # Inference
        outputs = inference_outlines(
            generator,
            dataset,
            args.batch_size,
            args.max_new_tokens,
            device,
            args.seed,
        )
    else:
        pipe = get_pipe(
            args.model,
            args.quantize_type,
            tokenizer,
            args.max_new_tokens,
            device,
        )
        start_time = get_start_time(device)

        # Inference
        outputs = inference(
            pipe,
            dataset,
            args.batch_size,
        )

    # Logging
    log_inference_time(start_time)
    log_memory_usage(device)

    # Evaluation
    labels = get_labels(args.template_file)
    # outputs = clean_pred(outputs, labels)
    scores = compute_scores(outputs, labels)

    # Save scores, args, and outputs
    out_file = get_out_file(args)
    write_result(scores, args, outputs, out_file)


def inference_outlines(
    generator: generate,
    dataset: Dataset,
    batch_size: int,
    max_new_tokens: int,
    device: str,
    seed: int,
) -> list[dict]:
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # Random number generator
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    preds = []
    for batch in tqdm(dataloader):
        preds_batch = generator(batch["prompt"], max_tokens=max_new_tokens, rng=rng)
        preds.extend(preds_batch)

    outputs = dataset.add_column("pred", preds).to_list()

    return outputs


def inference(
    pipe: pipeline,
    dataset: Dataset,
    batch_size: int,
    # temperature: float,
) -> list[dict]:
    outputs = []
    for record, res in tqdm(
        zip(
            dataset,
            pipe(
                KeyDataset(dataset, "prompt"),
                batch_size=batch_size,
                do_sample=False,
                # temperature=temperature,
            ),
        ),
        total=len(dataset),
    ):
        pred = res[0]["generated_text"].replace(record["prompt"], "").strip()
        record.update(pred=pred, gold=record["gold"])
        outputs.append(record)

    return outputs


if __name__ == "__main__":
    main()
