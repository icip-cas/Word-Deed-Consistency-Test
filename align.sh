#!/usr/bin/env bash
# -*- coding:utf-8 -*-

# Train Settings
model_names=('mistral-7b' 'mistral-7b-instruct' 'chatglm3-6b-base' 'chatglm3-6b')
name_or_paths=('/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/Mistral-7B-v0.1' '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/Mistral-7B-Instruct-v0.3' '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/chatglm-6b-base' '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/chatglm3-6b')
datasets=('WDCT_NE' 'WDCT_NE' 'WDCT_NE' 'WDCT_NE')
sft_lrs=('1e-5' '1e-6' '1e-5' '1e-5')
dpo_lrs=('5e-7' '5e-7' '5e-6' '5e-6')
train_modes=('sft' 'dpo')
changes=('speak' 'act')

# Train
for i in "${!model_names[@]}"; do
    model_name="${model_names[$i]}"
    name_or_path="${name_or_paths[$i]}"
    dataset="${datasets[$i]}"
    sft_lr="${sft_lrs[$i]}"
    dpo_lr="${dpo_lrs[$i]}"
    change="${changes[$i]}"

    # Infer
#    CUDA_VISIBLE_DEVICES=0,6 python eval/test_local_models.py \
#      --model ${model_name} \
#      --dataset ${dataset} \
#      --batch_size 32 \
#      --run_time 1

    # Train
    for train_mode in "${train_modes[@]}"
    do
      exp_name="${model_name}_${dataset}_${train_mode}_${change}"
      sft_model_path="output/${model_name}_${dataset}_sft_${change}/LATEST/policy.pt"
      model_path="output/${exp_name}/LATEST/policy.pt"
      result_path="result/${model_name}_${dataset}_${train_mode}_${change}.csv"
      log_file="log/${model_name}_${dataset}_${train_mode}_${change}.log"

      #模型训练
      if [ ${train_mode} = 'sft' ]; then
        CUDA_VISIBLE_DEVICES=0,6 python train.py \
          model=${model_name} \
          model.name_or_path=${name_or_path} \
          datasets="[${dataset}]" \
          change_mode=${change} \
          loss=${train_mode} \
          exp_name=${exp_name} \
          n_epochs=4 \
          gradient_accumulation_steps=1 \
          batch_size=2 \
          lr=${sft_lr} \
          warmup_steps=50 \
          eval_batch_size=32 \
          eval_every=6240 \
          minimum_log_interval_secs=1 \
          trainer=BasicTrainer \
          sample_during_eval=False 2>&1 | tee "$log_file"
      elif [ ${train_mode} = 'dpo' ]; then
        CUDA_VISIBLE_DEVICES=0,6 python train.py \
          model=${model_name} \
          model.name_or_path=${name_or_path} \
          model.archive=${sft_model_path} \
          datasets="[${dataset}]" \
          change_mode=${change} \
          loss=${train_mode} \
          loss.beta=0.1 \
          exp_name=${exp_name} \
          n_epochs=4 \
          gradient_accumulation_steps=1 \
          batch_size=8 \
          lr=${dpo_lr} \
          warmup_steps=50 \
          eval_batch_size=32 \
          eval_every=624 \
          minimum_log_interval_secs=1 \
          trainer=BasicTrainer \
          sample_during_eval=False 2>&1 | tee "$log_file"
      fi

      #训练后的模型推理
      CUDA_VISIBLE_DEVICES=0,6 python eval/test_local_models.py \
        --model ${model_name} \
        --model_path ${model_path} \
        --result_path ${result_path} \
        --dataset ${dataset} \
        --batch_size 64 \
        --run_time 1
    done
done