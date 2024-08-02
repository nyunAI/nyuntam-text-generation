export HF_DATASETS_CACHE="llama3_outputs/.cache"
export HF_HOME=$HF_DATASETS_CACHE

TARGET_MODEL="meta-llama/Meta-Llama-3-8B"                            # path or huggingface id of the base model. Used to access the tokenizer.
SEQLEN="2048"                                                                                  # model-specific maximal sequence length, 4096 for llama2, 8192 for mistral
DATASET="togethercomputer/RedPajama-Data-1T-Sample"                                                     # name/path of the dataset
OUTPUT_PATH="llama3_outputs/tokenized_datasets/llama3_togethercomputer_redpajama-data-1t-sample"         # path to save the tokenized dataset

rm -rf $OUTPUT_PATH/*

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=$HF_DATASETS_CACHE OMP_NUM_THREADS=16 torchrun \
    --master-port 3456 \
    --nproc-per-node=1 finetune_fsdp.py \
    --base_model $TARGET_MODEL \
    --quantized_model ./doesnt_matter \
    --load_dtype bfloat16 \
    --block_type LlamaDecoderLayer \
    --dataset_name=$DATASET \
    --split train \
    --cache_dir=$HF_DATASETS_CACHE \
    --trust_remote_code \
    --model_seqlen=$SEQLEN \
    --preprocessing_num_workers=64 \
    --save_dataset_and_exit $OUTPUT_PATH