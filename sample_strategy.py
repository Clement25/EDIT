"""Answer-selection strategies applied to one vLLM ``RequestOutput``."""
from utils import majority_answer


def prob_bon(output):
    """Probability best-of-N: return the sampled text with the highest cumulative log-probability.

    Requires ``SamplingParams(logprobs=...)`` so that ``cumulative_logprob`` is populated.
    """
    best = max(output.outputs, key=lambda o: o.cumulative_logprob if o.cumulative_logprob is not None else float('-inf'))
    return best.text


def sc(output):
    """Self-consistency: majority vote over the boxed answers of all samples.

    Returns a list (possibly empty) so that callers can use ``pred_num[-1]``.
    """
    ans = majority_answer([o.text for o in output.outputs])
    return [] if ans is None else [ans]
