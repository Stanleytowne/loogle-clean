#!/bin/bash

TASK="shortdep_qa"  # shortdep_qa, longdep_qa, longdep_summarization, shortdep_cloze
MAX_MODEL_LENGTH=32768
MAX_PROMPT_LENGTH=20000
MODEL_PATH=/ceph/home/muhan01/huggingfacemodels/Qwen2.5-3B-Instruct
OUTPUT_PATH=qwen2.5-3b.jsonl
RESULT_PATH=result.jsonl

TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
BATCH_SIZE=64

# Run inference
#python pred.py \
#    --task ${TASK} \
#    --max_model_length ${MAX_MODEL_LENGTH} \
#    --max_prompt_length ${MAX_PROMPT_LENGTH} \
#    --model_path ${MODEL_PATH} \
#    --output_path ${OUTPUT_PATH} \
#    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
#    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
#    --batch_size ${BATCH_SIZE}

python eval_qwen.py --loogle_file $OUTPUT_PATH --result_file $RESULT_PATH
