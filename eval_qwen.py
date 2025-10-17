"""
Standard LooGLE evaluation adopts GPT (OpenAI API).
However, we find Qwen2.5-32B-Instruct is sufficient for evaluation.
This script is used for debug, not final report.
"""
import argparse
import json
import numpy as np
import os
import tqdm
import torch
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = "You are a precise evaluator. Your task is to determine if the 'Predicted Answer' is semantically the same as the 'Ground Truth' for the given 'Question'. Your entire response MUST be only the single word 'True' or the single word 'False'. Do not provide any explanation or punctuation."
PROMPT_TEMPLATE = "Question: {question}\nGround Truth: {reference}\nPredicted Answer: {pred}"


def main():
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='/ceph/home/muhan01/huggingfacemodels/Qwen2.5-32B-Instruct', help="The local path of Qwen2.5-32B-Instruct.")
    parser.add_argument('--loogle_file', required=True, help="The path of the LooGLE output file. It should be a JSONL file.")
    parser.add_argument('--num_test', type=int, default=None, help="The number of entries to evaluate. If it is provided, the first `num_test` entries are evaluated; otherwise, all the entries are evaluated. (An entry refers to all the test QAs of an article, i.e., one line in the LooGLE output file.)")
    args = parser.parse_args()

    # Load the model
    llm = LLM(args.model_name_or_path, gpu_memory_utilization=0.95, dtype='bfloat16', max_model_len=32768)
    # llm = LLM(args.model_name_or_path, gpu_memory_utilization=0.95, dtype='bfloat16', max_model_len=8192, tensor_parallel_size=8)
    # llm = LLM(model=args.model_name_or_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, stop_token_ids=[151645, 151643])  # The generation config of Qwen2.5-32B-Instruct

    # Load the prediction file
    with open(args.loogle_file, 'r') as f:
        data = [json.loads(l) for l in f]
    if args.num_test is not None:
        data = data[:args.num_test]

    # Handle resuming
    all_scores = []  # For computing the average score

    # Eval
    all_msgs = []
    for entry in tqdm.tqdm(data):
        # Prepare messsages for batched eval
        msg = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': "user", 'content': PROMPT_TEMPLATE.format(question=entry['Q'], reference=entry['A'], pred=entry['P'])}
        ]
        all_msgs.append(msg)

    # Get responses
    all_responses = llm.chat(all_msgs, sampling_params, use_tqdm=False)
    all_results = []
    for qa, response in zip(data, all_responses):
        qa['score'] = 'true' in response.outputs[0].text.lower()
        all_results.append(qa)
        all_scores.append(qa['score'])

    # Report
    print(f"Avg. score = {np.mean(all_scores)}.")
    with open(args.loogle_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
