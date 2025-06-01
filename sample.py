import pandas as pd
import json
import argparse
import re
import os
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from time import time, sleep, perf_counter
from utils import get_saved_path, get_saved_path_dp
from sample_strategy import prob_bon, sc
from mapping import ds2hfaddrs, split2parquet, ds2anspattern, modelname2hfpath, modelname2pattern, \
    ds2querykeyname, ds2url
from vllm.utils import get_open_port
logging.basicConfig(level=logging.INFO)  # Ensure INFO messages are displayed


logging_step = 1

def load_dataset(dataset, split='test'):
    if dataset in ['math500']:
        df = pd.read_json(ds2hfaddrs[dataset] + split2parquet[dataset][split], lines=True)
    else:
        df = pd.read_parquet(ds2hfaddrs[dataset] + split2parquet[dataset][split])
    return df

prompt_template = """Below is a math question. Solve the question step by step, output the final answer in a box "\\boxed{{final_answer}}".

Question: {question}

Solution:
"""

def main_sample(args, dp_size, local_dp_rank, global_dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank):
    from vllm import LLM, SamplingParams
    # set up environment variables for distributed data parallel
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # load dataset
    logging.info('Start loading the dataset...')

    # prepare dataset
    if args.dataset.lower() in ['aimo']:
        split = 'train'
    elif 'train' in args.dataset.lower():
        dataset_name, split = args.dataset.split('-')
        args.dataset = dataset_name
        split = 'train'
    else:
        split = 'test'

    dataset = load_dataset(args.dataset, split)
    # prompts = dataset['question']
    # answers = dataset['answer']
    if args.max_examples > 0:
        dataset = dataset[:args.max_examples]

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    prompts_per_rank = (len(dataset) + dp_size - 1) // dp_size
    start = global_dp_rank * prompts_per_rank
    end = start + prompts_per_rank
    rank_dataset = dataset[start:end]


    # rename id
    id_key = 'id'
    if args.dataset == 'gsm8k':
        rank_dataset.reset_index(inplace=True)
        id_key = 'index'
    elif args.dataset == 'math500':
        rank_dataset['id'] = rank_dataset.apply(lambda row: row['unique_id'], axis=1)
        rank_dataset.drop('unique_id', axis=1, inplace=True)



    query_key_name = ds2querykeyname[args.dataset]
    rank_dataset['templated_prompts'] = rank_dataset.apply(lambda x: prompt_template.format(question=x[query_key_name]), axis=1)

    # vllm configs
    # set up sampling params according to the strategy
    if args.strategy is None:   # direct prompting
        args.n_sample = 1
        sampling_params = SamplingParams(temperature=0, max_tokens=args.max_token, n=args.n_samples)
    elif args.strategy == 'prob_bon':   # prob best-of-N
        sampling_params = SamplingParams(temperature=0.9, max_tokens=args.max_token, n=args.n_samples, logprobs=1)
    elif args.strategy == 'sc':
        sampling_params = SamplingParams(temperature=0.9, max_tokens=args.max_token, n=args.n_samples)
    elif args.strategy == 'all':    # preserving all N outputs
        sampling_params = SamplingParams(temperature=0.9, max_tokens=args.max_token, n=args.n_samples)
    else:
        raise NotImplementedError

    llm = LLM(
                model=modelname2hfpath[args.model],
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=GPUs_per_dp_rank,
            )
    tknzer = AutoTokenizer.from_pretrained(modelname2hfpath[args.model])

    # stats
    correct = 0
    evaluated = 0

    prompt_list = rank_dataset['templated_prompts'].to_list()
    templated_prompt_list = tknzer.apply_chat_template(
        [
            [
                {
                    'role': 'user',
                    'content': prompt
                }
            ] for prompt in prompt_list
        ], 
        tokenize=False
    )

    # stats
    correct = 0
    evaluated = 0

    start = perf_counter()
    full_outputs = llm.generate(templated_prompt_list, sampling_params)
    end = perf_counter()

    del llm

    result_list = []
    logging.info(f'Collect results on {global_dp_rank=}...')

    # reset index, otherwise it would overflow here
    with open(get_saved_path(args), 'w') as f:
        rank_dataset = rank_dataset.reset_index(drop=True)
        for i, row in rank_dataset.iterrows():
            prompt = row['templated_prompts']
            answer = row['answer']
            output = full_outputs[i]
            if args.strategy is None:
                output_solution = output.outputs[0].text
            elif args.strategy == 'prob_bon':
                output_solution = prob_bon(output)
            elif args.strategy == 'all':
                pred_num = []
                output_solution = [output.outputs[i].text for i in range(len(output.outputs))]
            else:
                raise NotImplementedError

            # only gsm8k needs to extract the last boxed answer
            if args.dataset == 'gsm8k':
                truth = re.findall(ds2anspattern[args.dataset], answer)[-1]
            else:
                truth = answer

            evaluated += 1
            if len(pred_num) > 0:
                correct += (pred_num[-1] == truth)

            result = {
                'question': prompt,
                'output_solution': output_solution,
                'pred_num': pred_num,
                'answer': answer,
                'answer_num': truth,
            }
            if args.dataset == 'math':
                result['level'] = row['level']

            result['id'] = row[id_key]
            result['question'] = prompt
            result['answer'] = answer
            f.write(json.dumps(result) + '\n')

    del rank_dataset


prompt_template_concise = """Below is a math question. Solve the question step by step, output the final answer in a box "\\boxed{{final_answer}}". You should organize your reasoning chains as short as possible.

Question: {question}

Solution:
"""

def main_sample_concise(args, dp_size, local_dp_rank, global_dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank):
    from vllm import LLM, SamplingParams
    # set up environment variables for distributed data parallel
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # load dataset
    logging.info('Start loading the dataset...')

    # prepare dataset
    if args.dataset.lower() in ['aimo']:
        split = 'train'
    elif 'train' in args.dataset.lower():
        dataset_name, split = args.dataset.split('-')
        args.dataset = dataset_name
        split = 'train'
    else:
        split = 'test'

    dataset = load_dataset(args.dataset, split)
    # prompts = dataset['question']
    # answers = dataset['answer']
    if args.max_examples > 0:
        dataset = dataset[:args.max_examples]

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    prompts_per_rank = (len(dataset) + dp_size - 1) // dp_size
    start = global_dp_rank * prompts_per_rank
    end = start + prompts_per_rank
    rank_dataset = dataset[start:end]

    # rename id
    id_key = 'id'
    if args.dataset == 'gsm8k':
        rank_dataset.reset_index(inplace=True)
        id_key = 'index'
    elif args.dataset == 'math500':
        rank_dataset['id'] = rank_dataset.apply(lambda row: row['unique_id'], axis=1)
        rank_dataset.drop('unique_id', axis=1, inplace=True)


    query_key_name = ds2querykeyname[args.dataset]

    rank_dataset['templated_prompts'] = rank_dataset.apply(lambda x: prompt_template_concise.format(question=x[query_key_name]), axis=1)

    # vllm configs
    # set up sampling params according to the strategy
    if args.strategy is None:   # direct prompting
        args.n_sample = 1
        sampling_params = SamplingParams(temperature=0, max_tokens=args.max_token, n=args.n_samples)
    elif args.strategy == 'prob_bon':   # prob best-of-N
        sampling_params = SamplingParams(temperature=0.9, max_tokens=args.max_token, n=args.n_samples, logprobs=1)
    elif args.strategy == 'sc':
        sampling_params = SamplingParams(temperature=0.9, max_tokens=args.max_token, n=args.n_samples)
    elif args.strategy == 'all':    # preserving all N outputs
        sampling_params = SamplingParams(temperature=0.9, max_tokens=args.max_token, n=args.n_samples)
    else:
        raise NotImplementedError

    llm = LLM(
                model=modelname2hfpath[args.model],
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=GPUs_per_dp_rank,
            )
    tknzer = AutoTokenizer.from_pretrained(modelname2hfpath[args.model])

    # stats
    correct = 0
    evaluated = 0

    prompt_list = rank_dataset['templated_prompts'].to_list()
    templated_prompt_list = tknzer.apply_chat_template(
        [
            [
                {
                    'role': 'user',
                    'content': prompt
                }
            ] for prompt in prompt_list
        ], 
        tokenize=False
    )

    # stats
    correct = 0
    evaluated = 0

    start = perf_counter()
    full_outputs = llm.generate(templated_prompt_list, sampling_params)
    end = perf_counter()

    del llm

    result_list = []
    logging.info(f'Collect results on {global_dp_rank=}...')

    # reset index, otherwise it would overflow here
    rank_dataset = rank_dataset.reset_index(drop=True)
    with open(get_saved_path(args), 'w') as f:
        for i, row in rank_dataset.iterrows():
            prompt = row['templated_prompts']
            answer = row['answer']
            output = full_outputs[i]
            if args.strategy is None:
                output_solution = output.outputs[0].text
            elif args.strategy == 'prob_bon':
                output_solution = prob_bon(output)
            elif args.strategy == 'all':
                pred_num = []
                output_solution = [output.outputs[i].text for i in range(len(output.outputs))]
            else:
                raise NotImplementedError


            # only gsm8k needs to extract the last boxed answer
            if args.dataset == 'gsm8k':
                truth = re.findall(ds2anspattern[args.dataset], answer)[-1]
            else:
                truth = answer

            evaluated += 1
            if len(pred_num) > 0:
                correct += (pred_num[-1] == truth)

            result = {
                'question': prompt,
                'output_solution': output_solution,
                'pred_num': pred_num,
                'answer': answer,
                'answer_num': truth,
            }
            if args.dataset == 'math':
                result['level'] = row['level']

            result_list.append(result)

            result = {
                'output_solution': output_solution,
                'pred_num': pred_num,
            }
            if args.dataset == 'math':
                result['level'] = row['level']

            # write to jsonl
            result['id'] = row[id_key]
            result['question'] = prompt
            result['answer'] = answer
            f.write(json.dumps(result) + '\n')

    del rank_dataset



#### Restricted Sampling

prompt_template_restrict = """Solve the given math problem step by step. You must output the final answer in a box "\\boxed{{final_answer}}". You are limited to at most {num_step} reasoning steps. Stop generating immediately and output the answer if you reach the maximum {num_step} steps or obtained the final answer early.

Question: {question}

Solution:
"""

def main_restrict(args, dp_size, local_dp_rank, global_dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank):
    from vllm import LLM, SamplingParams
    # set up environment variables for distributed data parallel
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # load dataset
    logging.info('Start loading the dataset...')

    # prepare dataset
    if args.dataset.lower() in ['aimo']:
        split = 'train'
    elif 'train' in args.dataset.lower():
        dataset_name, split = args.dataset.split('-')
        args.dataset = dataset_name
        split = 'train'
    else:
        split = 'test'
    dataset = load_dataset(args.dataset, split)
    # prompts = dataset['question']
    # answers = dataset['answer']
    if args.max_examples > 0:
        dataset = dataset[:args.max_examples]

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    prompts_per_rank = (len(dataset) + dp_size - 1) // dp_size
    start = global_dp_rank * prompts_per_rank
    end = start + prompts_per_rank
    rank_dataset = dataset[start:end]


    # rename id
    id_key = 'id'
    if args.dataset == 'gsm8k':
        rank_dataset.reset_index(inplace=True)
        id_key = 'index'
    elif args.dataset == 'math500':
        rank_dataset['id'] = rank_dataset.apply(lambda row: row['unique_id'], axis=1)
        rank_dataset.drop('unique_id', axis=1, inplace=True)
        

    query_key_name = ds2querykeyname[args.dataset]

    rank_dataset_list = []
    for i in range(16, -1, -1):
        new_rank_dataset = rank_dataset.copy()
        new_rank_dataset['templated_prompts'] = rank_dataset.apply(lambda x: prompt_template_restrict.format(question=x[query_key_name], num_step=i), axis=1)
        new_rank_dataset['max_steps'] = [i] * len(rank_dataset)
        rank_dataset_list.append(new_rank_dataset)
        
    rank_dataset = pd.concat(rank_dataset_list, ignore_index=True)

    # vllm configs
    # set up sampling params according to the strategy
    if args.strategy is None:   # direct prompting
        args.n_sample = 1
        sampling_params = SamplingParams(temperature=0, max_tokens=args.max_token, n=args.n_samples)
    elif args.strategy == 'prob_bon':   # prob best-of-N
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_token, n=args.n_samples, logprobs=1)
    elif args.strategy == 'sc':
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_token, n=args.n_samples)
    elif args.strategy == 'all':    # preserving all N outputs
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_token, n=args.n_samples, logprobs=0)
    else:
        raise NotImplementedError

    # initialize llm and tokenizer
    llm = LLM(
                model=modelname2hfpath[args.model],
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=GPUs_per_dp_rank,
            )
    tknz = AutoTokenizer.from_pretrained(modelname2hfpath[args.model])

    # stats
    correct, evaluated = 0, 0

    prompt_list = rank_dataset['templated_prompts'].to_list()
    templated_prompt_list = tknz.apply_chat_template(
        [
            [
                {   
                    'role': 'user',
                    'content': prompt
                }
            ] for prompt in prompt_list
        ], tokenize=False
    )

    start = perf_counter()
    full_outputs = llm.generate(templated_prompt_list, sampling_params)
    end = perf_counter()

    del llm

    result_dict = {}
    logging.info(f'Collect results on {global_dp_rank=}...')

    # reset index, otherwise it would overflow here
    with open(get_saved_path(args), 'w') as f:
        rank_dataset = rank_dataset.reset_index(drop=True)
        for i, row in rank_dataset.iterrows():
            prompt = row['templated_prompts']
            answer = row['answer']
            max_steps = row['max_steps']
            output = full_outputs[i]

            if args.strategy is None:
                output_solution = output.outputs[0].text
            elif args.strategy == 'prob_bon':
                output_solution = prob_bon(output)
            elif args.strategy == 'sc':
                pred_num = sc(output)
            elif args.strategy == 'all':
                pred_num = []
                output_solution = [output.outputs[i].text for i in range(len(output.outputs))]
            else:
                raise NotImplementedError
            
            cum_logprobs = [output.outputs[i].cumulative_logprob for i in range(len(output.outputs))]
            output_solution = [dict(text=output_solution[i], cumulative_logprob=cum_logprobs[0]) for i in range(len(output.outputs))]

            # only gsm8k needs to extract the last boxed answer
            if args.dataset == 'gsm8k':
                truth = re.findall(ds2anspattern[args.dataset], answer)[-1]
            else:
                truth = answer

            evaluated += 1
            if len(pred_num) > 0:
                correct += (pred_num[-1] == truth)

            result = {
                'output_solution': output_solution,
                'pred_num': pred_num,
                'max_steps': max_steps
            }
            if args.dataset == 'math':
                result['level'] = row['level']

            # if row[id_key] in result_dict:
            #     result_dict[row[id_key]]['result_list'].append(result)
            # else:
            #     result_dict[row[id_key]] = {
            #         'question': prompt,
            #         'result_list': [result],
            #         'answer_num': truth,
            #         'answer': answer
            #     }

            # write to jsonl
            result['id'] = row[id_key]
            result['question'] = prompt
            result['answer'] = answer
            f.write(json.dumps(result) + '\n')

    del rank_dataset



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", help="target dataset to be tested on")
    parser.add_argument("--model", default="qwmath7b", help="model to be evaluated")
    parser.add_argument("--output_dir", default='results/', help="model to be evaluated")
    parser.add_argument('--batch_size', type=int, default=256, help='maximum number of sequences per iteration') # no longer required, as VLLM handles batching internally
    parser.add_argument('--strategy', type=str, default=None, choices=[
        'prob_bon', '', 'sc', 'all'
    ], help='maximum number of sequences per iteration') 
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='targeted max GPU memory utilization') 
    parser.add_argument('--n_samples', type=int, default=1, help='number of sampled paths in each step') 
    parser.add_argument('--max_examples', type=int, default=-1, help='maximum number of sequences to be tested')

    parser.add_argument('--main_sample', action='store_true', help='whether to run main_sample')
    parser.add_argument('--main_concise', action='store_true', help='whether to run main_sample')
    parser.add_argument('--main_restrict', action='store_true', help='whether to run main_sample')

    # data parallel setting
    parser.add_argument("--dp-size",
                        type=int,
                        default=2,
                        help="Data parallel size")
    parser.add_argument("--tp-size",
                        type=int,
                        default=2,
                        help="Tensor parallel size")
    parser.add_argument("--node-size",
                        type=int,
                        default=1,
                        help="Total number of nodes")
    parser.add_argument("--node-rank",
                        type=int,
                        default=0,
                        help="Rank of the current node")
    parser.add_argument("--master-addr",
                        type=str,
                        default="",
                        help="Master node IP address")
    parser.add_argument("--master-port",
                        type=int,
                        default=0,
                        help="Master node port")
    parser.add_argument("--max_token",
                        type=int,
                        default=4096,
                        help="Max token length")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate the model")
    return parser

if __name__ == '__main__':
    from multiprocessing import Process, Queue
    parser = get_parser()
    args = parser.parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    procs = []
    if dp_per_node > 1:
        # current not support DP + sample_n
        raise NotImplementedError
    else:
        main_sample(args, dp_size, 0, 0, dp_master_ip, dp_master_port, tp_size)
        main_sample_concise(args, dp_size, 0, 0, dp_master_ip, dp_master_port, tp_size)

        args.n_samples = 16
        main_restrict(args, dp_size, 0, 0, dp_master_ip, dp_master_port, tp_size)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode
            # process results



    exit(exit_code)
