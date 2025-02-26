# Large Language Models Often Say One Thing and Do Another
- An implementation for Large Language Models Often Say One Thing and Do Another.
- Please contact @Ruoxi Xu (ruoxi2021@iscas.ac.cn) for questions and suggestions.

# Requirements

General
- Python (verified on 3.10)
- CUDA (verified on 12.1)

Python Packages

- see requirements.txt

```bash
conda create -n wdct python=3.10
conda activate wdct

pip install -r requirements.txt
```

# Quick Start

## Words and Deeds Consistency Test

The WDCT dataset is located in `data/WDCT.csv`. Each row represents a test case with the following columns:

- Type: Domain category of the test case. Possible values: Opinion, NonEthV (non ethical value), EthV (ethical value), Theory.
- id: Unique identifier for the test case.
- speak: A word question that probes a model's opinions, values, or related attributes through direct queries.
- act: A deed question that evaluates a model's hypothetical actions in grounded real-world scenarios.
- correct_answer: The ground-truth answer for domains with definitive answers.

## Model Evaluation

Before proceeding with the evaluation, please ensure that the model path is correctly specified in the `config.py` file.

To evaluate a single model, execute the following command:

```bash
python eval/test_local_models.py \
    --model llama-2-7b \
    --test_setting normal \
    --batch_size 32 \
    --temperature 0 \
    --max_new_tokens 10 \
    --run_time 1
```

- `--model` refers to the model name to be evaluated.
- `--test_setting` refers to the evaluation setting. You can choose between 'normal' and '3_shot_cot'.
- `--run_time` refers to the number of evaluation attempts. If multiple evaluations are required, this parameter can be adjusted.

Alternatively, if you need to evaluate multiple models, you can run the `eval.sh` script:

```bash
./eval.sh
```

## Model Alignment

This implementation aligns LLM's words or deeds on WDCT using the [DPO (Direct Preference Optimization) algorithm](https://arxiv.org/abs/2305.18290).

The DPO process consists of two stages:
- Supervised Fine-Tuning (SFT): Perform supervised fine-tuning on the dataset(s) of interest. This ensures that the preference data is in-distribution for the policy, providing a good foundation for the learning from preferences phase.
- Preference Learning: Use preference data to train the model from step 1.

To align a single model, run the following command:

```bash
# Infer
python eval/test_local_models.py \
    --model ${model_name} \
    --dataset ${dataset} \
    --batch_size 32 \
    --run_time 1

# SFT
python train.py \
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
   
# DPO
python train.py \
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
```

Alternatively, if you need to evaluate multiple models, you can run the `align.sh` script:

```bash
./align.sh
```

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.