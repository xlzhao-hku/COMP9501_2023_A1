import argparse
import pprint
import sys
import os
from typing import Iterable, Dict
from str2bool import str2bool


parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default="MATH")
parser.add_argument('--device_id', type=str, default="")
parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
parser.add_argument('--output_path', type=str, help="")
parser.add_argument('--start_index', type=int, default=0, help="")
parser.add_argument('--end_index', type=int, default=164, help="")
parser.add_argument('--temperature', type=float, default=0.8, help="")
parser.add_argument('--N', type=int, default=200, help="")
parser.add_argument('--max_len', type=int, default=512, help="")
parser.add_argument('--decoding_style', type=str, default='sampling', help="")
parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
parser.add_argument('--overwrite', action='store_true', default=False, help='')
parser.add_argument('--prompt_type', type=str, default="v1.0")
parser.add_argument('--greedy_decode', type=str2bool, default=False, help='')
parser.add_argument('--home_path', type=str, default="")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)


from tqdm import tqdm
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def get_math_problems(
        data_path="Data/MATH",
        split="test",
        data_category='algebra,counting_and_probability,geometry,intermediate_algebra,number_theory,prealgebra,precalculus',
):
    data_dir = os.path.join(data_path, split)
    dataset = {}
    for category in data_category.strip().split(","):
        files = sorted(os.listdir(os.path.join(data_dir, category)))
        for fid, file in enumerate(files):
            with open(os.path.join(data_dir, category, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            task_id = "{}_{}".format(category, fid)
            dataset[task_id] = {
                "prompt": data["problem"],
                "solution": data["solution"],
                "level": data["level"],
                "type": data["type"],
            }
    return dataset


def generate_prompt(input, prompt_type="v1.0"):
    if prompt_type == "v1.0":
        INSTRUCTION = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
import math
import numpy as np

def solve() -> float:
    \"\"\" {input}
    \"\"\"

### Response:"""
    else:
        raise NotImplementedError
    return INSTRUCTION.format_map(dict(input=input))


def get_model(
    load_8bit: bool = False,
    base_model: str = "bigcode/starcoder",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model


def main():

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.task == "MATH":
        problems = get_math_problems(data_path="{}/Data/MATH".format(args.home_path), split="test")

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids] # "task_id": {prompt: problem; response: answer; level: level}
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer, model = get_model(base_model=args.model)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False if args.greedy_decode else True,
        temperature=args.temperature,
        max_length=args.max_len,
        num_return_sequences=args.num_seqs_per_iter,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.95
    )

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t') # get problem
        prompt_batch = [generate_prompt(prompt, args.prompt_type)] # get full prompt

        ids_batch = [task_ids[i]]

        completion_seqs = []

        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)

        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                gen_tokens = model.generate(
                    **encoding,
                    generation_config=generation_config
                )

            if gen_tokens is not None:
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    if len(gen_seq.split("### Response:")) >= 2:
                        completion_seq = gen_seq.split("### Response:")[1]
                    else:
                        completion_seq = gen_seq
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {
                            'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                        }
                    )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()