export HF_DATASETS_CACHE="llama3_outputs/.cache"
export HF_HOME=$HF_DATASETS_CACHE

MODEL_PATH="meta-llama/Meta-Llama-3-8B"
CONVERTED_CHECKPOINT_PATH="llama3_outputs/converted_finetuned_quantized_model/meta-llama_meta-llama-3-8b" # path to the converted finetuned quantized model
QUANTIZED_MODEL_PATH="" # path to the quantized model
LOG_FILE="llama3_outputs/eval_logs_converted_finetuned_quantized_meta-llama_meta-llama-3-8b.log"

CUDA_VISIBLE_DEVICES=0,1,2,3
python lmeval.py \
    --model_path $MODEL_PATH \
    --cache_dir $HF_DATASETS_CACHE/hub \
    --converted_finetuned_quantized_model_ckpt $CONVERTED_CHECKPOINT_PATH \
    --log_file $LOG_FILE \
    --tasks winogrande:5 truthfulqa_mc2:0 gsm8k:5 mmlu:5 arc_challenge:25 hellaswag:10