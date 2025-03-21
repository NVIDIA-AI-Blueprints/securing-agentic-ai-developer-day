import os
import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import (
    HuggingFacePipelineCompatible,
    register_llm_provider,
)


def _get_model_config(config: RailsConfig, type: str):
    """Quick helper to return the config for a specific model type."""
    for model_config in config.models:
        if model_config.type == type:
            return model_config


def _load_model(model_name_or_path, device, num_gpus, hf_auth_token=None, debug=False):
    """Load an HF locally saved checkpoint."""
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update(
                    {
                        "device_map": "auto",
                        "max_memory": {i: "8GiB" for i in range(num_gpus)},
                    }
                )
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        print("mps not supported")
    else:
        raise ValueError(f"Invalid device: {device}")

    if hf_auth_token is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, **kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_auth_token=hf_auth_token, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            use_auth_token=hf_auth_token,
            **kwargs,
        )

    if device == "cuda" and num_gpus == 1:
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


def init_main_llm(config: RailsConfig):
    """Initialize the main model from a locally saved path.

    The path is taken from the main model config.

    models:
      - type: main
        engine: hf_pipeline_bloke
        parameters:
          path: "<PATH TO THE LOCALLY SAVED CHECKPOINT>"
    """
    model_config = _get_model_config(config, "main")
    model_path = model_config.parameters.get("path")
    device = model_config.parameters.get("device", "cuda")
    num_gpus = model_config.parameters.get("num_gpus", 1)
    hf_token = os.getenv("HF_TOKEN")
    model, tokenizer = _load_model(
        model_path, device, num_gpus, hf_auth_token=hf_token, debug=False
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
    )

    hf_llm = HuggingFacePipelineCompatible(pipeline=pipe)
    provider = get_llm_instance_wrapper(
        llm_instance=hf_llm, llm_type="hf_pipeline_llama3_8b"
    )
    register_llm_provider("hf_pipeline_llama3_8b", provider)


def init(llm_rails: LLMRails):
    config = llm_rails.config

    # Initialize the various models
    init_main_llm(config)
