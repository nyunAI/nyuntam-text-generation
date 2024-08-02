# export HF_DATASETS_CACHE="llama3_outputs/.cache"
# export HF_HOME=$HF_DATASETS_CACHE

# MODEL_PATH="meta-llama/Meta-Llama-3-8B"                                     # path or huggingface id of the base model
# DATASET_PATH="pajama"                                                             # name of the dataset
# MODEL_SEQLEN="4096"                                                               # model-specific maximal sequence length, 4096 for llama2, 8192 for mistral
# NBITS_PER_CODEBOOK="16"
# GROUP_SIZE="8"                                                                    
# BLOCKWISE_FINETUNE_EPOCHS="25"                                                    # set to 0 to disable blockwise finetuning during calibration
# CUDA_VISIBLE_DEVICES="0,1,2,3"                                                    # or e.g. 0,1,2,3
# SAVE_PATH="llama3_outputs/quantized_model/meta-llama_meta-llama-3-8b"         # path to save the quantized model

# python main.py \
#     $MODEL_PATH \
#     $DATASET_PATH \
#     --nsamples=2048 \
#     --val_size=256 \
#     --model_seqlen=4096 \
#     --num_codebooks=1 \
#     --nbits_per_codebook=$NBITS_PER_CODEBOOK \
#     --out_group_size=1 \
#     --in_group_size=$GROUP_SIZE \
#     --beam_size=1 \
#     --relative_mse_tolerance=0.01 \
#     --max_epochs=100 \
#     --finetune_lr=1e-4 \
#     --finetune_adam_beta1=0.90 \
#     --finetune_adam_beta2=0.999 \
#     --finetune_keep_best \
#     --finetune_batch_size=64 \
#     --local_batch_size=4 \
#     --finetune_max_epochs=$BLOCKWISE_FINETUNE_EPOCHS \
#     --finetune_early_stop=3 \
#     --offload_activations \
#     --trust_remote_code \
#     --save $SAVE_PATH \
#     --resume




export HF_DATASETS_CACHE="llama3_outputs/.cache"
export HF_HOME=$HF_DATASETS_CACHE

MODEL_PATH="{MODEL_PATH}"                                     
DATASET_PATH="{DATASET_PATH}"                                                             
MODEL_SEQLEN="{model_seqlen}"                                                               
NBITS_PER_CODEBOOK="{nbits_per_codebook}"
GROUP_SIZE="{in_group_size}"                                                                    
BLOCKWISE_FINETUNE_EPOCHS="{finetune_max_epochs}"                                                    
CUDA_VISIBLE_DEVICES="0,1,2,3"                                                    
SAVE_PATH="{save}"        

python main.py \
    $MODEL_PATH \
    $DATASET_PATH \
    --nsamples={nsamples} \
    --val_size={val_size} \
    --model_seqlen={model_seqlen} \
    --num_codebooks={num_codebooks} \
    --nbits_per_codebook={nbits_per_codebook} \
    --out_group_size={out_group_size} \
    --in_group_size={in_group_size} \
    --beam_size={beam_size} \
    --relative_mse_tolerance={relative_mse_tolerance} \
    --max_epochs={max_epochs} \
    --finetune_lr={finetune_lr} \
    --finetune_adam_beta1={finetune_adam_beta1} \
    --finetune_adam_beta2={finetune_adam_beta2} \
    --finetune_keep_best \
    --finetune_batch_size={finetune_batch_size} \
    --local_batch_size={local_batch_size} \
    --finetune_max_epochs={finetune_max_epochs} \
    --finetune_early_stop={finetune_early_stop} \
    --offload_activations \
    --trust_remote_code \
    --save={save} \
    --resume