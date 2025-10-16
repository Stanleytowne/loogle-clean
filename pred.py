import os
import torch
import json
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None, help="long context understanding tasks in LooGLE", choices=["shortdep_qa","longdep_qa","longdep_summarization","shortdep_cloze"])
    parser.add_argument('--max_prompt_length', type=int, default=None, help="the max length of input prompt")
    parser.add_argument('--max_model_length', type=int, default=None, help="the max length of model input")

    parser.add_argument('--model_path', type=str, default="./Models/") 
    parser.add_argument('--output_path', type=str, default="./Output/")
    
    # vLLM related parameters
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="number of GPUs to use for tensor parallelism")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="GPU memory utilization for vLLM (0.0-1.0)")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size for inference")

    return parser.parse_args(args)



def prepare_prompts(data_instances, tokenizer, max_length, prompt_format):
    """Prepare all prompts and return metadata"""
    all_prompts = []
    metadata = []  # Store metadata for each prompt
    
    for data_idx, data_instance in enumerate(data_instances):
        raw_inputs = data_instance['input']

        qa_list = eval(data_instance['qa_pairs'])
        for qa_idx, qa_pair in enumerate(qa_list):
            json_obj = {'Q': qa_pair['Q'], 'input': raw_inputs}
            prompt = prompt_format.format(**json_obj)
            
            # Handle long prompts
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            all_prompts.append(formatted_prompt)
            metadata.append({
                'qa_idx': qa_idx,
                'Q': qa_pair['Q'],
                'A': qa_pair['A']
            })
    
    return all_prompts, metadata


def batch_generate(llm, prompts, max_gen, batch_size=8):
    """Batch generation with support for splitting large batches"""
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=max_gen,
    )
    
    all_outputs = []
    # Process in batches to avoid memory overflow
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(outputs)
    
    return all_outputs


def aggregate_results(data_instances, outputs, metadata):
    """Aggregate generated results back to original data structure"""
    results = []
    
    for output, meta in zip(outputs, metadata):
        results.append({
            'Q': meta['Q'],
            'A': meta['A'],
            'P': output.outputs[0].text
        })

    return results

def loads(path, task):
    data = []
    with open(path+task+".jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

if __name__ == '__main__':

    args = parse_args()

    # data = load_dataset('bigainlco/LooGLE', args.task, split="test")
    data = loads("LooGLE-testdata/", args.task)

    # Load model using vLLM
    print(f"Loading model: {args.model_path}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Still need tokenizer for prompt truncation
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    task2prompt = json.load(open("./config/task2prompt.json", "r"))
    task2maxlen = json.load(open("./config/task2maxlen.json", "r"))
    prompt_format = task2prompt[args.task]
    max_gen = task2maxlen[args.task]

    print(f"Total {len(data)} data instances, using batch size: {args.batch_size}")
    
    # Process all data in batches
    for batch_start in range(0, len(data), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//args.batch_size + 1}/{(len(data)-1)//args.batch_size + 1} (data {batch_start+1}-{batch_end}/{len(data)})")
        
        # Prepare prompts
        prompts, metadata = prepare_prompts(batch_data, tokenizer, args.max_prompt_length, prompt_format)
        print(f"  Generated {len(prompts)} prompts")
        
        # Batch inference
        outputs = batch_generate(llm, prompts, max_gen, batch_size=args.batch_size)
        
        # Aggregate results
        results = aggregate_results(batch_data, outputs, metadata)
        
        # Write results
        with open(args.output_path, "a+") as g:
            for preds in results:
                g.write(json.dumps(preds)+'\n')
        
        print(f"  Completed batch {batch_start//args.batch_size + 1}")
    
    print("All data processing completed!")


