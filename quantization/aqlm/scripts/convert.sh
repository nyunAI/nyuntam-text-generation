export HF_DATASETS_CACHE="llama3_outputs/.cache"
export HF_HOME=$HF_DATASETS_CACHE

ORIG_MODEL_PATH="meta-llama/Meta-Llama-3-8B"
MODEL_PATH="llama3_outputs/fine_tuned/meta-llama_meta-llama-3-8b"
CONVERTED_CHECKPOINT_PATH="llama3_outputs/converted_finetuned_quantized_model/meta-llama_meta-llama-3-8b"

rm -rf $CONVERTED_CHECKPOINT_PATH/*
python convert_legacy_model_format.py\
    --monkeypatch_old_pickle \
    --base_model $ORIG_MODEL_PATH\
    --finetune_fsdp_dir $MODEL_PATH\
    --code_dtype int32 \
    --load_dtype auto \
    --save $CONVERTED_CHECKPOINT_PATH
