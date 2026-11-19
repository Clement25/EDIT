"""Generate reasoning chains with vLLM.

Three passes share one model load:

* ``sample``   - standard step-by-step prompt
* ``concise``  - step-by-step prompt + "keep the reasoning as short as possible"
* ``restrict`` - step-constrained prompt swept over num_step = max_step..0
                 (the sample pool consumed by edit_search.py)

Each pass writes ``<output_dir>/<dataset>_<model>_<pass>.jsonl``.
"""
import argparse
import glob
import json
import logging
import os
import re
from time import perf_counter

import pandas as pd

from mapping import ds2hfaddrs, split2parquet, ds2anspattern, modelname2hfpath, ds2querykeyname
from sample_strategy import prob_bon, sc
from utils import get_saved_path, get_saved_path_dp, majority_answer, normalize_answer

logging.basicConfig(level=logging.INFO)


def load_dataset(dataset, split='test'):
    if dataset not in ds2hfaddrs:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {sorted(ds2hfaddrs)}")
    if split not in split2parquet[dataset]:
        raise ValueError(f"Dataset '{dataset}' has no '{split}' split. Available: {sorted(split2parquet[dataset])}")
    path = ds2hfaddrs[dataset] + split2parquet[dataset][split]
    if path.endswith('.jsonl'):
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


prompt_template = """Below is a math question. Solve the question step by step, output the final answer in a box "\\boxed{{final_answer}}".

Question: {question}

Solution:
"""

prompt_template_concise = """Below is a math question. Solve the question step by step, output the final answer in a box "\\boxed{{final_answer}}". You should organize your reasoning chains as short as possible.

Question: {question}

Solution:
"""

prompt_template_restrict = """Solve the given math problem step by step. You must output the final answer in a box "\\boxed{{final_answer}}". You are limited to at most {num_step} reasoning steps. Stop generating immediately and output the answer if you reach the maximum {num_step} steps or obtained the final answer early.

Question: {question}

Solution:
"""

PASSES = ('sample', 'concise', 'restrict')


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def resolve_dataset(args):
    """Return (dataset_name, split). Mutates nothing."""
    name = args.dataset
    if name.lower() == 'aimo':
        return 'aimo', 'train'
    if name.lower().endswith('-train'):
        return name[:-len('-train')], 'train'
    return name, 'test'


def prepare_rank_dataset(args, dataset_name, split, dp_size, global_dp_rank):
    dataset = load_dataset(dataset_name, split)
    if args.max_examples > 0:
        dataset = dataset[:args.max_examples]

    # every rank takes a contiguous slice of the dataset
    per_rank = (len(dataset) + dp_size - 1) // dp_size
    start = global_dp_rank * per_rank
    rank_dataset = dataset[start:start + per_rank].copy()

    if dataset_name == 'gsm8k':
        rank_dataset = rank_dataset.reset_index()
        rank_dataset = rank_dataset.rename(columns={'index': 'id'})
    elif dataset_name == 'math500':
        rank_dataset = rank_dataset.rename(columns={'unique_id': 'id'})
    return rank_dataset.reset_index(drop=True)


def gold_answer(dataset_name, answer):
    if dataset_name in ds2anspattern:
        return re.findall(ds2anspattern[dataset_name], answer)[-1]
    return answer


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def make_sampling_params(args, n):
    from vllm import SamplingParams
    if args.strategy is None:  # greedy direct prompting
        return SamplingParams(temperature=0, max_tokens=args.max_token, n=1)
    kwargs = dict(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_token, n=n)
    if args.strategy in ('prob_bon', 'all'):
        kwargs['logprobs'] = 0  # populates cumulative_logprob
    return SamplingParams(**kwargs)


def collect(output, strategy):
    """Turn one RequestOutput into (output_solution, pred_num)."""
    texts = [o.text for o in output.outputs]
    if strategy is None:
        return texts[0], [a for a in [majority_answer(texts[:1])] if a is not None]
    if strategy == 'prob_bon':
        sol = prob_bon(output)
        return sol, [a for a in [majority_answer([sol])] if a is not None]
    if strategy == 'sc':
        return texts, sc(output)
    if strategy == 'all':
        sols = [dict(text=o.text, cumulative_logprob=o.cumulative_logprob) for o in output.outputs]
        return sols, [a for a in [majority_answer(texts)] if a is not None]
    raise NotImplementedError(strategy)


def run_pass(args, mode, llm, tokenizer, dataset_name, rank_dataset, out_path):
    query_key = ds2querykeyname[dataset_name]

    if mode == 'restrict':
        frames = []
        for k in range(args.max_step, -1, -1):
            f = rank_dataset.copy()
            f['templated_prompts'] = f[query_key].apply(lambda q: prompt_template_restrict.format(question=q, num_step=k))
            f['max_steps'] = k
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
        n = args.restrict_samples
    else:
        template = prompt_template if mode == 'sample' else prompt_template_concise
        df = rank_dataset.copy()
        df['templated_prompts'] = df[query_key].apply(lambda q: template.format(question=q))
        n = args.n_samples

    chat_prompts = tokenizer.apply_chat_template(
        [[{'role': 'user', 'content': p}] for p in df['templated_prompts']],
        tokenize=False, add_generation_prompt=True,
    )

    t0 = perf_counter()
    outputs = llm.generate(chat_prompts, make_sampling_params(args, n))
    logging.info(f'[{mode}] generated {len(outputs)} prompts in {perf_counter() - t0:.1f}s')

    correct = 0
    with open(out_path, 'w') as f:
        for i, row in df.iterrows():
            output_solution, pred_num = collect(outputs[i], args.strategy)
            truth = gold_answer(dataset_name, row['answer'])
            if pred_num and normalize_answer(pred_num[-1]) == normalize_answer(truth):
                correct += 1
            result = {
                'id': row['id'],
                'question': row['templated_prompts'],
                'output_solution': output_solution,
                'pred_num': pred_num,
                'answer': row['answer'],
                'answer_num': truth,
            }
            if mode == 'restrict':
                result['max_steps'] = int(row['max_steps'])
            if 'level' in row:
                result['level'] = row['level']
            f.write(json.dumps(result, default=str) + '\n')

    logging.info(f'[{mode}] majority-vote accuracy on this shard: {correct}/{len(df)} prompts')


def worker(args, passes, dp_size, local_dp_rank, global_dp_rank, dp_master_ip, dp_master_port, tp_size):
    # vLLM data-parallel environment
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    from vllm import LLM
    from transformers import AutoTokenizer

    dataset_name, split = resolve_dataset(args)
    rank_dataset = prepare_rank_dataset(args, dataset_name, split, dp_size, global_dp_rank)
    logging.info(f'rank {global_dp_rank}: {len(rank_dataset)} examples from {dataset_name}/{split}')

    model_path = modelname2hfpath[args.model]
    llm = LLM(model=model_path, gpu_memory_utilization=args.gpu_memory_utilization, tensor_parallel_size=tp_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for mode in passes:
        pass_args = argparse.Namespace(**{**vars(args), 'dataset': dataset_name, 'mode': mode})
        out_path = get_saved_path_dp(pass_args, global_dp_rank) if dp_size > 1 else get_saved_path(pass_args)
        run_pass(args, mode, llm, tokenizer, dataset_name, rank_dataset, out_path)


def merge_shards(args, passes):
    """Concatenate per-rank shards written on this node into one file per pass."""
    dataset_name, _ = resolve_dataset(args)
    for mode in passes:
        pass_args = argparse.Namespace(**{**vars(args), 'dataset': dataset_name, 'mode': mode})
        final = get_saved_path(pass_args)
        shards = sorted(glob.glob(final[:-len('.jsonl')] + '.rank*.jsonl.part'),
                        key=lambda p: int(re.search(r'\.rank(\d+)\.', p).group(1)))
        if not shards:
            continue
        with open(final, 'w') as out:
            for s in shards:
                with open(s) as inp:
                    out.write(inp.read())
                os.remove(s)
        logging.info(f'merged {len(shards)} shards into {final}')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", help="gsm8k | math500 | aimo (append -train for a training split)")
    parser.add_argument("--model", default="qwmath7b", help="model key from mapping.modelname2hfpath")
    parser.add_argument("--output_dir", default='results/edit', help="directory for result jsonl files")
    parser.add_argument('--strategy', type=str, default='all', choices=['none', 'prob_bon', 'sc', 'all'],
                        help="'all' keeps every sample (needed by edit_search.py); 'none' = greedy decoding")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--n_samples', type=int, default=1, help='samples per prompt in the sample / concise passes')
    parser.add_argument('--restrict_samples', type=int, default=16, help='samples per prompt and per step limit in the restrict pass')
    parser.add_argument('--max_step', type=int, default=16, help='restrict pass sweeps num_step = max_step, ..., 0')
    parser.add_argument('--max_examples', type=int, default=-1, help='only use the first k examples (debugging)')
    parser.add_argument("--max_token", type=int, default=4096, help="max generated tokens")

    # which passes to run; none given = all three
    parser.add_argument('--main_sample', action='store_true', help='run the standard-prompt pass')
    parser.add_argument('--main_concise', action='store_true', help='run the concise-prompt pass')
    parser.add_argument('--main_restrict', action='store_true', help='run the step-constrained pass')

    # parallelism
    parser.add_argument("--dp-size", type=int, default=1, help="data parallel size (total across nodes)")
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size per DP rank")
    parser.add_argument("--node-size", type=int, default=1, help="total number of nodes")
    parser.add_argument("--node-rank", type=int, default=0, help="rank of the current node")
    parser.add_argument("--master-addr", type=str, default="", help="master node IP (multi-node only)")
    parser.add_argument("--master-port", type=int, default=0, help="master node port (multi-node only)")
    return parser


def main():
    args = get_parser().parse_args()
    if args.strategy == 'none':
        args.strategy = None
    if args.model not in modelname2hfpath:
        raise ValueError(f"Unknown model '{args.model}'. Available: {sorted(modelname2hfpath)}")

    selected = [m for m, flag in zip(PASSES, (args.main_sample, args.main_concise, args.main_restrict)) if flag]
    passes = selected or list(PASSES)

    dp_size, node_size, node_rank = args.dp_size, args.node_size, args.node_rank
    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    if node_size == 1:
        from vllm.utils import get_open_port
        dp_master_ip, dp_master_port = "127.0.0.1", get_open_port()
    else:
        dp_master_ip, dp_master_port = args.master_addr, args.master_port

    if dp_size == 1:
        worker(args, passes, 1, 0, 0, dp_master_ip, dp_master_port, args.tp_size)
        return 0

    from multiprocessing import get_context
    ctx = get_context('spawn')
    procs = []
    for local_rank in range(dp_per_node):
        global_rank = node_rank * dp_per_node + local_rank
        p = ctx.Process(target=worker, args=(args, passes, dp_size, local_rank, global_rank,
                                              dp_master_ip, dp_master_port, args.tp_size))
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode:
            exit_code = p.exitcode

    if exit_code == 0 and node_size == 1:
        merge_shards(args, passes)
    elif node_size > 1:
        logging.info('multi-node run: per-rank shards are left in place; concatenate them once all nodes finish')
    return exit_code


if __name__ == '__main__':
    exit(main())
