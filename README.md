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
| `sample.py` | Generates reasoning chains with vLLM in three passes (below) and writes one JSONL file per pass. |
| `edit_search.py` | Runs the dual-goal search (Algorithm 1) offline on the step-constrained samples and reports accuracy and response length. |
| `mapping.py` | Dataset and model registries (Hugging Face paths, question/answer columns). |
| `utils.py` | `\boxed{}` answer extraction, answer normalisation, output paths. |
| `sample_strategy.py` | Answer-selection strategies (`prob_bon`, `sc`). |
| `view_jsonl_results.ipynb` | Per-step-limit majority-vote analysis of the restrict pass, with outputs truncated to a token budget (2048 / 16384). |

`sample.py` loads the model once and runs three passes. Pass the matching flag to run only some of them; with no flag all three run.

| Pass | Flag | Prompt |
|---|---|---|
| `sample` | `--main_sample` | Standard step-by-step prompt. |
| `concise` | `--main_concise` | Same prompt, plus an instruction to keep the reasoning as short as possible. |
| `restrict` | `--main_restrict` | Step-constrained prompt above, swept over `num_step` = 16, 15, …, 0 with 16 samples per limit. Each record stores `max_steps` and, per sample, the text and cumulative log-probability. |

Results go to `<output_dir>/<dataset>_<model>_<pass>.jsonl`.

## Setup

```bash
pip install vllm transformers pandas huggingface_hub
```

Datasets are read straight from the Hugging Face Hub. Experiments in the paper ran on 4× NVIDIA A100 80GB with tensor parallelism.

## Usage

**1. Sample.** Use the `restrict` pass for EDIT:

```bash
python sample.py --dataset math500 --model qwen3-8b --main_restrict \
    --tp-size 4 --max_token 2048
```

To use more GPUs, add data parallelism: `--dp-size 2 --tp-size 2` runs two vLLM replicas and merges their outputs. For multi-node runs, add `--node-size`, `--node-rank`, `--master-addr` and `--master-port`; shards are then left per rank for you to concatenate.

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `gsm8k` | `gsm8k`, `math500` or `aimo`. Append `-train` for the GSM8K training split. |
| `--model` | `qwmath7b` | Key in `mapping.modelname2hfpath`. Also available: `llama3.1-8b-instruct`, `qwen3-8b`, `qwen3-32b`, `qwq-32b`, `phi4-reasoning`, `r1-distill-qwen1.5b`, `r1-distill-llama8b-instruct`. |
| `--strategy` | `all` | `all` keeps every sample, which `edit_search.py` needs. `sc` is self-consistency, `prob_bon` is highest log-probability, `none` is greedy decoding. |
| `--temperature` / `--top_p` | `0.7` / `0.9` | Sampling settings from the paper. |
| `--restrict_samples` | `16` | Samples per prompt and per step limit in the restrict pass. |
| `--max_step` | `16` | The restrict pass sweeps `num_step` from this value down to 0. |
| `--n_samples` | `1` | Samples per prompt in the sample / concise passes. |
| `--max_token` | `4096` | Max generated tokens. The paper uses 2048 for GSM8K / MATH500 and 16384 for AIMO. |
| `--dp-size` / `--tp-size` | `1` / `1` | Data-parallel replicas / tensor-parallel GPUs per replica. |
| `--output_dir` | `results/edit` | Where result files are written. |
| `--max_examples` | `-1` | Use only the first *k* examples (for debugging). |

**2. Search.**

```bash
python edit_search.py results/edit/math500_qwen3-8b_restrict.jsonl --model qwen3-8b \
    --budget 64 --max_iter 6 --patience 1 --output results/edit/math500_qwen3-8b_edit.jsonl
```

This prints accuracy and the average token length of the returned responses. With `--output` it also writes, per question, the chosen answer, the response and the full search trace. Omit `--model` to measure length in characters and avoid loading a tokenizer.

Because the search is replayed on pre-sampled outputs, each iteration draws `budget // max_iter` unseen samples at the current step limit. Where Algorithm 1 leaves details open, `edit_search.py` makes these choices:

- τ is the integer step limit, starting at τ_max = 16 and τ_min = 0. Midpoints are rounded down.
- The consistency check passes when the majority answer matches the previous iteration's and the length statistic drops. Pass `--no_length_drop` to compare answers only.
- `most_confident(H)` sums each answer's votes over all iterations, breaking ties by the shorter length statistic.
- The returned response is the shortest sample, among all samples drawn, whose answer matches the final answer.

**3. Analyse (optional).** Open `view_jsonl_results.ipynb`, set `folder` to your output directory under `results/`, and run the cells for the per-step-limit analysis.

## Citation

```bibtex
@article{han2025long2short,
  title   = {From Long to Short: LLMs Excel at Trimming Own Reasoning Chains},
  author  = {Han, Wei and Zhan, Geng and Yu, Sicheng and Wang, Chenyu and Hooi, Bryan},
  journal = {arXiv preprint arXiv:2509.06174},
  year    = {2025}
}
```
