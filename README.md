# EDIT: Efficient Dynamic Inference Trimming

Official code for **[From Long to Short: LLMs Excel at Trimming Own Reasoning Chains](https://arxiv.org/abs/2509.06174)**

Wei Han, Geng Zhan, Sicheng Yu, Chenyu Wang, Bryan Hooi

*NeurIPS 2025 Workshop on Efficient Reasoning*

[![arXiv](https://img.shields.io/badge/arXiv-2509.06174-b31b1b.svg)](https://arxiv.org/abs/2509.06174)

---

## Overview

Large reasoning models (LRMs) often *overthink*: they produce far longer reasoning chains than a problem needs. **EDIT** is a training-free, test-time scaling method that lets a model trim its own reasoning. Model weights are untouched; EDIT only changes how candidates are generated and selected.

The method rests on two observations:

1. **Length follows the constraint.** If the prompt limits the model to at most *k* reasoning steps, the output-length distribution shifts with *k*.
2. **Too tight hurts correctness.** Under stringent constraints, the model starts favouring brevity over correctness, and wrong answers dominate the shortest outputs.

EDIT therefore searches for the **tightest constraint that still yields a reliable answer**.

## Method

**Constraint injection.** Each question is wrapped in a prompt that caps the number of reasoning steps:

```
Solve the given math problem step by step. You must output the final answer in a box "\boxed{final_answer}".
You are limited to at most {num_step} reasoning steps. Stop generating immediately and output the answer
if you reach the maximum {num_step} steps or obtained the final answer early.
```

**Dual-goal search.** Given a sample budget *N*, EDIT runs a binary search with patience over the constraint strength, spending *N/T* samples per iteration. At each step it tracks two signals:

- **Answer confidence**: the majority-vote answer among the samples.
- **Length statistic**: the mean of the minimum, lower quartile (Q1) and median length of the responses that give the majority answer.

If the majority answer stays consistent and the length responds to the tighter constraint, the search keeps tightening. If not, a patience counter allows a few more attempts before the lower bound is relaxed so the model can reason more freely. The final output is the most confident answer seen during the search. See Algorithm 1 in the paper for full details.

## Results

EDIT is compared with Direct CoT prompting (greedy), Best-of-N (majority vote, then shortest) and Self-Truncation, using N = 64 samples per question. EDIT keeps accuracy close to the strongest baseline (mostly within 1 to 2 points) while producing substantially shorter reasoning.

EDIT accuracy (%) / average output length (tokens), from Table 2 of the paper:

| Model | GSM8K | MATH500 | AIMO |
|---|---|---|---|
| Llama-3.1-8B | 71.4 / 87 | 52.4 / 108 | 11.1 / 40 |
| Qwen3-8B | 93.4 / 103 | 73.2 / 207 | 74.4 / 2284 |
| Qwen3-32B | 94.4 / 83 | 70.8 / 297 | 83.3 / 4140 |
| QwQ-32B | 93.3 / 223 | 64.8 / 298 | 86.7 / 3791 |
| Phi-4-Reasoning | 91.8 / 286 | 80.2 / 335 | 83.3 / 2533 |
| R1-Distill-Qwen-1.5B | 85.7 / 154 | 62.4 / 249 | 41.1 / 1376 |
| R1-Distill-Llama-3.1-8B | 88.7 / 262 | 65.0 / 337 | 60.0 / 1770 |

AIMO is 90 problems from AIME 2022–2024. See the paper for baseline numbers, the penalized-length and accuracy-per-effective-length (AEU) metrics, and ablations.

## Repository contents

| File | Purpose |
|---|---|
| `sample.py` | Generates reasoning chains with vLLM under three prompting modes (see below) and writes them to JSONL. |
| `view_jsonl_results.ipynb` | Loads the JSONL results, truncates each sample to a token budget (2048 / 16384), majority-votes the `\boxed{}` answers and prints accuracy per model. |

`sample.py` runs three passes in sequence:

1. **`main_sample`**: standard step-by-step prompt.
2. **`main_sample_concise`**: the same prompt plus an instruction to keep the reasoning as short as possible.
3. **`main_restrict`**: the step-constrained prompt above, swept over `num_step` = 16, 15, …, 0 with 16 samples per constraint. Each record stores `max_steps` and, per sample, the text and cumulative log-probability. This is the raw material for the dual-goal search.

> **Note:** `sample.py` and the notebook import helper modules (`utils.py`, `sample_strategy.py`, `mapping.py`) that hold the dataset and model registries, the answer extractor and the output-path logic. These are not yet in this repository.

## Setup

```bash
pip install vllm transformers pandas tqdm
```

Experiments in the paper ran on 4× NVIDIA A100 80GB with tensor parallelism.

## Usage

Generate samples (single node; data parallelism across processes is not supported yet, so set `--dp-size 1`):

```bash
python sample.py \
    --dataset math500 \
    --model <model_key> \
    --strategy all \
    --dp-size 1 \
    --tp-size 4 \
    --max_token 2048
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `gsm8k` | `gsm8k`, `math500`, `math` or `aimo`. Append `-train` to use a training split. |
| `--model` | `qwmath7b` | Model key, resolved to a Hugging Face path via `mapping.py`. |
| `--strategy` | `None` | `all` keeps every sample (used for EDIT). `prob_bon` and `sc` are also defined. |
| `--n_samples` | `1` | Samples per prompt for the first two passes; the constrained pass always uses 16. |
| `--max_token` | `4096` | Max generated tokens. The paper uses 2048 for GSM8K / MATH500 and 16384 for AIMO. |
| `--tp-size` | `2` | Tensor-parallel size passed to vLLM. |
| `--gpu_memory_utilization` | `0.95` | vLLM GPU memory fraction. |
| `--max_examples` | `-1` | Evaluate only the first *k* examples (for debugging). |

Then open `view_jsonl_results.ipynb`, set `folder` and the glob to where your results were written (the notebook expects `results/<folder>/<dataset>_<model>_*.jsonl`) and run the cells.

## Citation

```bibtex
@article{han2025long2short,
  title   = {From Long to Short: LLMs Excel at Trimming Own Reasoning Chains},
  author  = {Han, Wei and Zhan, Geng and Yu, Sicheng and Wang, Chenyu and Hooi, Bryan},
  journal = {arXiv preprint arXiv:2509.06174},
  year    = {2025}
}
```
