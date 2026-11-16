"""Dual-goal search (Algorithm 1 of the EDIT paper), run offline.

The ``restrict`` pass of sample.py stores, for every question and every step
limit ``num_step = max_step..0``, a pool of samples. This script replays the
search on that pool: each iteration draws ``n = N // T`` unseen samples at the
current step limit tau_t instead of calling the model, so a single sampling
run can be searched with different hyper-parameters.

Where the paper leaves a detail open, the choice made here is exposed as an
option (see ``EditConfig``):

* tau is the integer step limit; the search starts with tau_min=0, tau_max=max_step
  and midpoints are rounded down.
* "the length statistic responds to the constraint" is read as
  lstat_t < lstat_{t-1} after tightening (``require_length_drop``).
* most_confident(H) sums the votes each answer received over all iterations;
  ties are broken by the smaller length statistic.
* When patience runs out, the bound is moved and the next tau is the midpoint of
  the new bounds; the iteration is still recorded in H (``record_on_reset``).
* The returned response is the shortest sampled response whose answer equals
  the final answer; its length is what the efficiency numbers report.

Usage:
    python edit_search.py results/edit/math500_qwen3-8b_restrict.jsonl --model qwen3-8b
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Dict, List, Optional

from utils import extract_last_boxed, normalize_answer


@dataclass
class EditConfig:
    budget: int = 64            # N: total samples per question
    max_iter: int = 6           # T
    patience: int = 1           # beta_0
    tau_min: int = 0
    tau_max: Optional[int] = None   # default: largest step limit in the pool
    require_length_drop: bool = True
    record_on_reset: bool = True
    seed: int = 0


@dataclass
class Sample:
    text: str
    answer: Optional[str]
    length: int


@dataclass
class SearchResult:
    answer: Optional[str]
    response: Optional[Sample]
    trace: List[dict] = field(default_factory=list)
    samples_used: int = 0


def q1(values):
    """Lower quartile (median of the lower half, inclusive method)."""
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    return median(v[: (len(v) + 1) // 2])


def length_stat(samples: List[Sample], answer) -> float:
    """Eq. 4: mean of min, Q1 and median length over responses giving ``answer``."""
    lens = [s.length for s in samples if s.answer == answer]
    if not lens:
        return float('inf')
    return (min(lens) + q1(lens) + median(lens)) / 3


def vote(samples: List[Sample]):
    cnt = Counter(s.answer for s in samples if s.answer is not None)
    if not cnt:
        return None, 0
    return cnt.most_common(1)[0]


def most_confident(history: List[dict]):
    if not history:
        return None
    votes, best_len = defaultdict(int), {}
    for h in history:
        if h['answer'] is None:
            continue
        votes[h['answer']] += h['votes']
        best_len[h['answer']] = min(best_len.get(h['answer'], float('inf')), h['lstat'])
    if not votes:
        return None
    return max(votes, key=lambda a: (votes[a], -best_len[a]))


def dual_goal_search(pool: Dict[int, List[Sample]], cfg: EditConfig) -> SearchResult:
    """Run Algorithm 1 for one question. ``pool`` maps step limit -> samples."""
    rng = random.Random(cfg.seed)
    remaining = {k: rng.sample(v, len(v)) for k, v in pool.items()}
    levels = sorted(pool)

    def nearest_level(tau):
        return min(levels, key=lambda k: (abs(k - tau), -k))

    n = max(1, cfg.budget // cfg.max_iter)
    tau_min = cfg.tau_min
    tau_max = cfg.tau_max if cfg.tau_max is not None else max(levels)
    tau_t = tau_max
    beta = cfg.patience
    history: List[dict] = []
    seen: List[Sample] = []

    for t in range(cfg.max_iter):
        level = nearest_level(tau_t)
        drawn = remaining[level][:n]
        remaining[level] = remaining[level][n:]
        if not drawn:  # pool at this level exhausted
            drawn = rng.sample(pool[level], min(n, len(pool[level])))
        seen.extend(drawn)

        ans, votes = vote(drawn)
        lstat = length_stat(drawn, ans)
        entry = dict(iter=t, tau=level, answer=ans, votes=votes, n=len(drawn), lstat=lstat)

        if history:
            prev = history[-1]
            same = ans is not None and ans == prev['answer']
            shorter = lstat < prev['lstat'] if cfg.require_length_drop else True
            consistent = same and shorter
        else:
            consistent = ans is not None

        record = True
        if consistent:
            entry['action'] = 'tighten'
            tau_max = level
            tau_t = (tau_max + tau_min) // 2
        elif beta == 0:
            if ans is not None and ans == most_confident(history):
                entry['action'] = 'reset:tighten'
                tau_max = level
            else:
                entry['action'] = 'reset:loosen'
                tau_min = level
            beta = cfg.patience
            tau_t = (tau_max + tau_min) // 2
            record = cfg.record_on_reset
        else:
            entry['action'] = 'patience'
            tau_t = (level + tau_min) // 2
            beta -= 1

        entry['next_tau'] = tau_t
        if record:
            history.append(entry)

    final = most_confident(history)
    if final is None:  # nothing recorded with a parsable answer: vote over everything seen
        final, _ = vote(seen)
    matching = [s for s in seen if s.answer == final and final is not None]
    response = min(matching, key=lambda s: s.length) if matching else None
    return SearchResult(answer=final, response=response, trace=history, samples_used=len(seen))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_pools(path: str, length_fn: Callable[[str], int]):
    """Group a restrict-pass jsonl into {id: (gold, {max_steps: [Sample, ...]})}."""
    pools, golds = defaultdict(lambda: defaultdict(list)), {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = row['id']
            golds[qid] = row.get('answer_num', row['answer'])
            sols = row['output_solution']
            sols = sols if isinstance(sols, list) else [sols]
            for s in sols:
                text = s['text'] if isinstance(s, dict) else s
                pools[qid][int(row['max_steps'])].append(
                    Sample(text=text, answer=normalize_answer(extract_last_boxed(text)), length=length_fn(text)))
    return {qid: (golds[qid], dict(pool)) for qid, pool in pools.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('result_file', help='jsonl written by the restrict pass of sample.py')
    p.add_argument('--model', help='model key (mapping.py) for token lengths; omit to count characters')
    p.add_argument('--budget', type=int, default=64, help='N, total samples per question')
    p.add_argument('--max_iter', type=int, default=6, help='T, number of search iterations')
    p.add_argument('--patience', type=int, default=1, help='beta_0')
    p.add_argument('--no_length_drop', action='store_true', help='consistency check uses answers only')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', help='optional jsonl with per-question answer, response and search trace')
    args = p.parse_args()

    if args.model:
        from transformers import AutoTokenizer
        from mapping import modelname2hfpath
        tok = AutoTokenizer.from_pretrained(modelname2hfpath[args.model])
        length_fn = lambda s: len(tok.encode(s, add_special_tokens=False))
    else:
        length_fn = len

    cfg = EditConfig(budget=args.budget, max_iter=args.max_iter, patience=args.patience,
                     require_length_drop=not args.no_length_drop, seed=args.seed)
    pools = load_pools(args.result_file, length_fn)

    correct, total_len, n_resp = 0, 0, 0
    out = open(args.output, 'w') if args.output else None
    for qid, (gold, pool) in pools.items():
        res = dual_goal_search(pool, cfg)
        ok = res.answer is not None and res.answer == normalize_answer(gold)
        correct += ok
        if res.response is not None:
            total_len += res.response.length
            n_resp += 1
        if out:
            out.write(json.dumps(dict(id=qid, gold=gold, pred=res.answer, correct=bool(ok),
                                      response=res.response.text if res.response else None,
                                      length=res.response.length if res.response else None,
                                      samples_used=res.samples_used, trace=res.trace), default=str) + '\n')
    if out:
        out.close()

    unit = 'tokens' if args.model else 'chars'
    print(f'questions: {len(pools)}')
    print(f'accuracy: {correct / max(1, len(pools)) * 100:.2f}%')
    print(f'avg length of returned response: {total_len / max(1, n_resp):.1f} {unit}')


if __name__ == '__main__':
    main()
