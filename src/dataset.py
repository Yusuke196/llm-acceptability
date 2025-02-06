import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Generator

import pandas as pd
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.utils import read_jsonl, replace_words

# For inference.py and pred_by_yes_prob.py
# class PromptData:


def get_dataset_with_prompts(
    data_name: str,
    template_file: str,
    examples_file: str | None,
    tokenizer: AutoTokenizer,  # Used to build prompts for Llama-3, Mixtral, etc.
    model_name: str,
    debug: bool,
    rm_cache: bool = False,
) -> Dataset:
    in_dir = get_in_dir(data_name)
    system_prompt, user_template = get_templates(template_file, examples_file)

    if re.search(r"Instruct|Chat", model_name):
        assert (
            "chat" in template_file
        ), "An instruct model is used with a non-chat template"

    if rm_cache:
        remove_cache()

    if data_name.startswith("blimp") or data_name.startswith("climp"):
        gen = gen_data_with_prompts

    dataset = Dataset.from_generator(
        gen,
        gen_kwargs={
            "in_dir": in_dir,
            "system_prompt": system_prompt,
            "user_template": user_template,
            "template_file": template_file,
            "tokenizer": tokenizer,
            "model_name": model_name,
        },
    )
    if debug:
        dataset = dataset.select(range(10))

    return dataset


def get_in_dir(data_name: str) -> str:
    # TODO: make it possible to pass file_stems, which is a list like ['ga_wa'], to use specific data files
    if data_name == "ja_fluency":
        return "data/ja_fluency"
    elif data_name == "blimp":
        return "data/blimp/data"
    elif data_name == "climp":
        return "data/climp/CLiMP_corpus"
    elif data_name == "sandbox":
        return "data/sandbox"
    else:
        raise ValueError(f"Unknown data_name: {data_name}")


def get_templates(template_file: str, examples_file: str | None) -> tuple[str, str]:
    """
    - Get `system_prompt` and return it.
    - Fill the {examples} placeholder of the template, leaving the {sentence} placeholder. Return it as `template`.
    """
    templates_dict = json.load(open(template_file))

    system_prompt = templates_dict["system_prompt"]
    template = templates_dict["user_template"]

    # For few-shot templates
    if examples_file is None:
        examples_str = ""
    else:
        example_template = templates_dict["example_template"]
        examples = json.load(open(examples_file))
        examples_str = "".join(
            example_template.format(**example) for example in examples
        )

    # Note that format() cannot be used here because we should leave {sentence} as is
    template = template.replace("{examples}", examples_str)

    return system_prompt, template


def gen_data_with_prompts(
    in_dir: str,
    system_prompt: str,
    user_template: str,
    template_file: str,
    tokenizer: AutoTokenizer,
    model_name: str,
) -> Generator[dict, None, None] | Generator[tuple[dict, dict], None, None]:
    acceptable_keys_in_result = [
        "sentence_good",
        "sentence_bad",
        "gold",
    ]

    if "blimp" in in_dir:
        acceptable_keys_in_result += [
            "UID",
            "field",
        ]
        file_pattern = "*.jsonl"
        reader = read_jsonl
    elif "climp" in in_dir:
        acceptable_keys_in_result += [
            "paradigm",
            "phenomenon",
        ]
        file_pattern = "*.csv"
        reader = read_climp

    if "_ab" in template_file:
        acceptable_keys_in_result += [
            "system_prompt",
            "user_prompt",
            "prompt",
        ]
    elif "_yn" in template_file:
        acceptable_keys_in_result += [
            "system_prompt_good",
            "user_prompt_good",
            "prompt_good",
            "system_prompt_bad",
            "user_prompt_bad",
            "prompt_bad",
        ]

    for in_file in Path(in_dir).glob(file_pattern):
        for obj in reader(in_file):
            if "_ab" in template_file:
                # Randomly choose the place of good sentence
                alphabet_for_good, alphabet_for_bad = random.sample(["a", "b"], 2)
                obj[f"sentence_{alphabet_for_good}"] = obj["sentence_good"]
                obj[f"sentence_{alphabet_for_bad}"] = obj["sentence_bad"]
                # We make the model to predict the good sentence
                obj["gold"] = alphabet_for_good.upper()

                dct = _get_prompts(
                    obj,
                    system_prompt,
                    user_template,
                    template_file,
                    tokenizer,
                    model_name,
                )
                obj.update(dct)

            elif "_yn" in template_file:
                # Add prompts for each of the good sentence and the bad sentence
                # Later we will compare the yes-probabilities of the two prompts
                for good_or_bad in ["good", "bad"]:
                    obj["sentence"] = obj[f"sentence_{good_or_bad}"]
                    key_suffix = "_" + good_or_bad

                    dct = _get_prompts(
                        obj,
                        system_prompt,
                        user_template,
                        template_file,
                        tokenizer,
                        model_name,
                        key_suffix,
                    )
                    obj.update(dct)

            # Delete unnecessary keys
            res = {k: v for k, v in obj.items() if k in acceptable_keys_in_result}

            yield res


def _get_prompts(
    obj: dict,
    system_prompt: str,
    user_template: str,
    template_file: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    key_suffix: str = "",  # For "_yn" templates
) -> dict:
    res = {}
    user_prompt = user_template.format(**obj)

    if "blimp" in template_file:
        segmenter = " "
    elif "climp" in template_file or "_ja" in template_file:
        segmenter = ""
    else:
        raise ValueError(f"Unknown template_file: {template_file}")

    if "ja-ja" in template_file:
        mapping = {"Yes": "はい", "No": "いいえ"}
        user_prompt = replace_words(user_prompt, mapping)
        res["gold"] = replace_words(obj["gold"], mapping)

    if re.match(r"gpt|claude", model_name):
        res["system_prompt" + key_suffix] = system_prompt
        res["user_prompt" + key_suffix] = user_prompt
    elif re.search(r"Llama-3|Qwen|Mixtral|Yi-1.5|Smaug", model_name):
        if re.search(r"Instruct|Chat", model_name):
            res["prompt" + key_suffix] = get_prompt_chat(
                system_prompt, user_prompt, tokenizer, model_name
            )
        else:
            res["prompt" + key_suffix] = system_prompt + segmenter + user_prompt
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return res


def get_prompt_chat(
    system_prompt: str, user_prompt: str, tokenizer: AutoTokenizer, model_name: str
) -> str:
    if re.search(r"Llama-3|Qwen", model_name):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": system_prompt + " " + user_prompt},
        ]

    prompt = tokenizer.apply_chat_template(
        # add_generation_prompt=True ensures that when the model generates text it will write a bot response instead of doing something unexpected, like continuing the user’s message
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


# For pred_by_prob.py
# class ReadoutData:


def get_dataset(
    data_name: str,
    approach: str,
    tokenizer: AutoTokenizer,
    template_file: str = None,
    debug: bool = False,
    rm_cache: bool = False,
) -> Dataset:
    """
    Get a dataset for LL-based prediction.
    """
    in_dir = get_in_dir(data_name)

    if template_file is not None:
        template = open(template_file).read()
    else:
        template = None

    if rm_cache:
        remove_cache()

    dataset = Dataset.from_generator(
        gen_data,
        gen_kwargs={
            "in_dir": in_dir,
            "approach": approach,
            "template": template,
            "tokenizer": tokenizer,
            "debug": debug,
        },
    )
    if debug:
        dataset = dataset.select(range(10))

    return dataset


def gen_data(
    in_dir: str,
    approach: str,
    template: str | None,
    tokenizer: AutoTokenizer,
    debug: bool = False,
) -> Generator[dict, None, None]:
    if approach.startswith("ll_intemplate"):
        # `template_parts` should look like: ['Among the two sentences below, only A is grammatically correct.\n\nA: ', '{sentence}', '\nB: ', '{other_sentence}']
        template_parts = [
            s
            for s in re.split(r"(\{sentence\}|\{other_sentence\})", template)
            if s != ""
        ]
    elif approach in ["ll", "ll_normalized", "ll_pen"]:
        template_parts = ["{sentence}"]
    else:
        raise ValueError(f"Unknown approach: {approach}")

    if "blimp" in in_dir:
        if debug:
            file_pattern = "adjunct_island.jsonl"
        else:
            file_pattern = "*.jsonl"

        for in_file in Path(in_dir).glob(file_pattern):
            for obj in read_jsonl(in_file):
                obj = {
                    key: obj[key]
                    for key in [
                        "sentence_good",
                        "sentence_bad",
                        "field",
                        "UID",
                    ]
                }

                res = _preprocess_sentence(obj, template_parts, tokenizer)
                obj.update(res)

                yield obj

    elif "climp" in in_dir:
        if debug:
            file_pattern = "anaphor_agreement_gender_1000.csv"
        else:
            file_pattern = "*.csv"

        for in_file in Path(in_dir).glob(file_pattern):
            for obj in read_climp(in_file):
                # Below codes are the same as blimp
                res = _preprocess_sentence(obj, template_parts, tokenizer)
                obj.update(res)

                yield obj


def _preprocess_sentence(obj: dict, template_parts: list[str], tokenizer) -> dict:
    res = defaultdict(list)

    for good_or_bad in ["good", "bad"]:
        for template_part in template_parts:
            if template_part == "{sentence}":
                ids_part = tokenizer.encode(
                    obj[f"sentence_{good_or_bad}"], add_special_tokens=False
                )
                mask_int = 1
                # dd[f"ids_{good_or_bad}"] += ids_part
                # dd[f"mask_{good_or_bad}"] += [1] * len(ids_part)
            elif template_part == "{other_sentence}":
                if good_or_bad == "good":
                    good_or_bad_other = "bad"
                else:
                    good_or_bad_other = "good"
                ids_part = tokenizer.encode(
                    obj[f"sentence_{good_or_bad_other}"], add_special_tokens=False
                )
                mask_int = 1
                # dd[f"ids_{good_or_bad}"] += ids_part
                # dd[f"mask_{good_or_bad}"] += [1] * len(ids_part)
            else:
                ids_part = tokenizer.encode(template_part, add_special_tokens=False)
                mask_int = 0
                # dd[f"ids_{good_or_bad}"] += ids_part
                # dd[f"mask_{good_or_bad}"] += [0] * len(ids_part)

            res[f"ids_{good_or_bad}"] += ids_part
            res[f"mask_{good_or_bad}"] += [mask_int] * len(ids_part)

        # Add BOS token if the model has it
        if re.search(r"Llama-3|Mixtral", tokenizer.name_or_path):
            # We can't convert these into tensors here because Dataset object doesn't accept tensors seemingly
            res[f"ids_{good_or_bad}"] = [tokenizer.bos_token_id] + res[
                f"ids_{good_or_bad}"
            ]
            res[f"mask_{good_or_bad}"] = [0] + res[f"mask_{good_or_bad}"]

        # Update sentence_(good|bad)
        res[f"sentence_{good_or_bad}"] = tokenizer.decode(res[f"ids_{good_or_bad}"])

    return res


def get_dataloader(
    dataset: Dataset, batch_size: int, pad_token_id: int, data_name: str
) -> DataLoader:
    def collate_fn(batch: list[dict]):
        # Reformat existing items
        res = {
            "sentence_good": [item["sentence_good"] for item in batch],
            "sentence_bad": [item["sentence_bad"] for item in batch],
        }
        if data_name.startswith("blimp"):
            res.update({"UID": [item["UID"] for item in batch]})
        elif data_name.startswith("climp"):
            res.update(
                {
                    "paradigm": [item["paradigm"] for item in batch],
                    "phenomenon": [item["phenomenon"] for item in batch],
                }
            )

        ids_good = [torch.tensor(item["ids_good"]) for item in batch]
        mask_good = [torch.tensor(item["mask_good"]) for item in batch]
        ids_bad = [torch.tensor(item["ids_bad"]) for item in batch]
        mask_bad = [torch.tensor(item["mask_bad"]) for item in batch]
        res.update(
            {
                "ids_good": pad_sequence(
                    ids_good, batch_first=True, padding_value=pad_token_id
                ),
                "mask_good": pad_sequence(mask_good, batch_first=True, padding_value=0),
                "ids_bad": pad_sequence(
                    ids_bad, batch_first=True, padding_value=pad_token_id
                ),
                "mask_bad": pad_sequence(mask_bad, batch_first=True, padding_value=0),
            }
        )

        return res

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    return dataloader


# General


def read_climp(in_file: Path) -> Generator[dict, None, None]:
    df = pd.read_csv(
        in_file,
        names=["phenomenon", "paradigm", "sentence", "is_acceptable"],
        skiprows=1,
    )

    for i in range(0, len(df) - 1, 2):
        res = {
            "sentence_good": df.loc[i, "sentence"] + "。",
            "sentence_bad": df.loc[i + 1, "sentence"] + "。",
            "paradigm": df.loc[i, "paradigm"],
            "phenomenon": df.loc[i, "phenomenon"],
        }

        yield res


def remove_cache():
    path = "~/.cache/huggingface/datasets/generator/"
    full_path = os.path.expanduser(path)

    if os.path.exists(full_path):
        shutil.rmtree(full_path, ignore_errors=True)
