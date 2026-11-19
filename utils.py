import os
import re
from collections import Counter


def _result_stem(args):
    mode = getattr(args, 'mode', None) or 'sample'
    return f"{args.dataset}_{args.model}_{mode}"


def get_saved_path(args):
    """Path of the merged result file for the current pass.

    Files are named ``<dataset>_<model>_<pass>.jsonl``; the notebook relies on
    the first two ``_``-separated fields being the dataset and model keys.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    return os.path.join(args.output_dir, _result_stem(args) + '.jsonl')


def get_saved_path_dp(args, global_dp_rank):
    """Per-rank shard written by one data-parallel worker (merged afterwards)."""
    os.makedirs(args.output_dir, exist_ok=True)
    return os.path.join(args.output_dir, f"{_result_stem(args)}.rank{global_dp_rank}.jsonl.part")


def extract_last_boxed(text):
    r"""Return the content of the last ``\boxed{...}`` in ``text`` (nested braces
    handled), stripped of surrounding whitespace. Returns ``None`` if absent."""
    if not isinstance(text, str):
        return None
    idx = text.rfind('\\boxed')
    while idx != -1:
        j = idx + len('\\boxed')
        while j < len(text) and text[j] == ' ':
            j += 1
        if j < len(text) and text[j] == '{':
            depth, start = 0, j + 1
            for k in range(j, len(text)):
                if text[k] == '{':
                    depth += 1
                elif text[k] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:k].strip()
            return None  # unbalanced braces: treat as no answer
        idx = text.rfind('\\boxed', 0, idx)
    return None


def normalize_answer(ans):
    """Light normalisation so that e.g. '1,000', '1000.0' and ' 1000 ' compare equal."""
    if ans is None:
        return None
    s = str(ans).strip()
    s = s.replace('\\!', '').replace('$', '').strip()
    s = re.sub(r'^\\text\{(.*)\}$', r'\1', s)
    num = s.replace(',', '')
    try:
        f = float(num)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def majority_answer(texts):
    """Majority vote over the boxed answers of ``texts``; ``None`` if no answer parses."""
    cnt = Counter(a for a in (normalize_answer(extract_last_boxed(t)) for t in texts) if a is not None)
    if not cnt:
        return None
    return cnt.most_common(1)[0][0]
