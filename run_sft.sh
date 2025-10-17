#!/bin/bash

TASK="shortdep_qa_no_context"  # shortdep_qa, longdep_qa, longdep_summarization, shortdep_cloze, shortdep_qa_no_context
MAX_MODEL_LENGTH=32768
MAX_PROMPT_LENGTH=20000
MODEL_PATH=/ceph/home/muhan01/tpz/Long-Digestor-Experiments/outputs/stage1-1e-4/global_step_70
OUTPUT_PATH=output/stage1-1e-4.jsonl

TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
device=0

# Run inference
CUDA_VISIBLE_DEVICES=$device \
python pred.py \
    --task ${TASK} \
    --max_model_length ${MAX_MODEL_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION}

CUDA_VISIBLE_DEVICES=$device \
python eval_qwen.py --loogle_file $OUTPUT_PATH
