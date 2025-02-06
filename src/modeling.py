import re

import outlines
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    set_seed,
)


def get_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer_kwargs = {"padding_side": "left"}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def get_pipe(
    model_name: str,
    quantize_type: str,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    device: str = "auto",
) -> pipeline:
    model = get_model(model_name, quantize_type, tokenizer, device)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    return pipe


def get_model(
    model_name: str, quantize_type: str, tokenizer: AutoTokenizer, device: str = "auto"
) -> AutoModelForCausalLM:
    model_kwargs = _get_model_kwargs(model_name, quantize_type, tokenizer, device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    return model


def _get_model_kwargs(
    model_name: str, quantize_type: str, tokenizer: AutoTokenizer, device: str
):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
        "trust_remote_code": True,
        "do_sample": False,
    }

    if quantize_type == "4bit":
        # See https://huggingface.co/blog/4bit-transformers-bitsandbytes#advanced-usage
        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            }
        )
    elif quantize_type == "8bit":
        model_kwargs.update(
            {
                "quantization_config": {"load_in_8bit": True},
            }
        )
    elif quantize_type == "none":
        pass
    else:
        raise ValueError(f"Unknown quantize_type: {quantize_type}")

    if re.match(r"Llama-3.+Instruct", model_name):
        # https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct
        model_kwargs.update(
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )

    return model_kwargs


def get_generator(
    model_name: str,
    quantize_type: str,
    tokenizer: AutoTokenizer,
    device: str,
):
    model_kwargs = _get_model_kwargs(model_name, quantize_type, tokenizer, device)

    model = outlines.models.transformers(
        model_name,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"trust_remote_code": True, "device_map": device},
    )
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    model.model.eval()
    outlines.disable_cache()

    pattern = r"[AB]"
    sampler = outlines.samplers.greedy()
    generator = outlines.generate.regex(model, pattern, sampler=sampler)

    return generator
