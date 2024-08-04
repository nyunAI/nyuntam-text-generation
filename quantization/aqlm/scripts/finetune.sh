export HF_DATASETS_CACHE="llama3_outputs/.cache"
export HF_HOME=$HF_DATASETS_CACHE

MODEL_PATH="meta-llama/Meta-Llama-3-8B"  # path or huggingface id of the base model
QUANTIZED_MODEL_PATH="llama3_outputs/quantized_model/meta-llama_meta-llama-3-8b"
TOKENIZED_DATASET_PATH="llama3_outputs/tokenized_datasets/llama3_togethercomputer_redpajama-data-1t-sample"    # model-specific maximal sequence length, 4096 for llama2, 8192 for mistral
CACHE_DIR="llama3_outputs/.cache"
SAVE_PATH="llama3_outputs/fine_tuned/meta-llama_meta-llama-3-8b"
SEQLEN=2048


NUM_GPUS=4
rm -rf $SAVE_PATH
FSDP_CPU_RAM_EFFICIENT_LOADING=1 && torchrun --nproc-per-node=$NUM_GPUS finetune_fsdp.py \
    --base_model $MODEL_PATH --quantized_model $QUANTIZED_MODEL_PATH \
    --model_seqlen=$SEQLEN --block_type LlamaDecoderLayer --limit_parallel_inits 4 \
    --load_dtype float32 --amp_dtype float32 --code_dtype uint16 \
    --straight_through_buffer_dtype float32 \
    --dataset_name=$TOKENIZED_DATASET_PATH --split none --seed 1337 \
    --preprocessing_chunk_length 100000 --cache_dir=$CACHE_DIR --trust_remote_code \
    --update_codes --update_codebooks_and_scales --update_non_quantized_parameters \
    --lamb --debias --lr 1e-4 --adam_beta1 0.9 --adam_beta2 0.95 \
    --code_lr 1e-3 --code_beta1 0.0 --code_beta2 0.95 --beam_size 1 --delta_decay 0 \
    --max_code_change_per_step 1e-2 --code_trust_ratio 1e-2 --code_selection_temperature 0 \
    --batch_size=256 --microbatch_size=4 --max_epochs 10 --gradient_checkpointing \
    --print_every_steps=1 --verbose_optimizer  --eval_every_steps=1 \
    --save $SAVE_PATH --save_every_steps 1