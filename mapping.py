"""Registries for datasets and models used by sample.py and the analysis notebook.

Model keys must not contain underscores: result files are named
``<dataset>_<model>_<pass>.jsonl`` and the notebook recovers the model key by
splitting the filename on ``_``.
"""

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# Base location of each dataset, read with pandas through the ``hf://`` filesystem
# (requires ``huggingface_hub``).
ds2hfaddrs = {
    'gsm8k': 'hf://datasets/openai/gsm8k/',
    'math500': 'hf://datasets/HuggingFaceH4/MATH-500/',
    # AIME 2022-2024, 90 problems (the "AIMO" set in the paper)
    'aimo': 'hf://datasets/AI-MO/aimo-validation-aime/',
}

# File (relative to ``ds2hfaddrs``) for each split.
split2parquet = {
    'gsm8k': {
        'train': 'main/train-00000-of-00001.parquet',
        'test': 'main/test-00000-of-00001.parquet',
    },
    'math500': {
        'test': 'test.jsonl',
    },
    'aimo': {
        'train': 'data/train-00000-of-00001.parquet',
    },
}

# Column holding the question text.
ds2querykeyname = {
    'gsm8k': 'question',
    'math500': 'problem',
    'aimo': 'problem',
}

# Regex used to pull the gold answer out of the reference solution (GSM8K only;
# the other datasets store the final answer directly in the ``answer`` column).
ds2anspattern = {
    'gsm8k': r"#### (-?[\d,]*\.?\d+)",
}

ds2url = {
    'gsm8k': 'https://huggingface.co/datasets/openai/gsm8k',
    'math500': 'https://huggingface.co/datasets/HuggingFaceH4/MATH-500',
    'aimo': 'https://huggingface.co/datasets/AI-MO/aimo-validation-aime',
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

modelname2hfpath = {
    # instruction-tuned
    'llama3.1-8b-instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'qwmath7b': 'Qwen/Qwen2.5-Math-7B-Instruct',
    # RL-enhanced reasoning models
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'qwen3-32b': 'Qwen/Qwen3-32B',
    'qwq-32b': 'Qwen/QwQ-32B',
    'phi4-reasoning': 'microsoft/Phi-4-reasoning',
    # distilled from DeepSeek-R1
    'r1-distill-qwen1.5b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'r1-distill-llama8b-instruct': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
}

# All prompts ask for the answer in \boxed{...}; answers are parsed with
# utils.extract_last_boxed, so every model shares the same pattern.
modelname2pattern = {name: r"\\boxed\{" for name in modelname2hfpath}
