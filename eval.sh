# 定义颜色
GREEN='\033[0;32m'
NC='\033[0m'  # No Color

# 模型
models=('llama-2-7b' 'llama-2-7b-chat' 'llama-3-8b-instruct' 'mistral-7b' 'mistral-7b-instruct' 'chatglm3-6b-base' 'chatglm3-6b')

# 定义最大GPU数
max_gpus=8

# 固定参数
dataset='WDCT'
test_setting='normal'

# 测试
pids=()
for model in "${models[@]}"
do
  echo -e "${GREEN}Start infer: $model on $dataset${NC}"
  # 如果没有可用的GPU，等待进程完成
  while [ ${#pids[@]} -ge $max_gpus ]; do
    echo "Start waiting"
    wait -n "${pids[@]}"

    # 移除已完成的进程PID
    echo "Before removing, pids: ${pids[@]}"
    running_pids=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        running_pids+=("$pid")
      fi
    done
    pids=("${running_pids[@]}")
    echo "After removing, pids: ${pids[@]}"
  done

  # 获取可用的GPU索引
  gpu_index=$(nvidia-smi --query-gpu=index,gpu_name,utilization.gpu,utilization.memory,memory.free --format=csv,noheader,nounits | awk -F ', ' '{if($3 == 0 && $4 == 0) print $1}' | head -n 1)
  echo "GPU that can be used: $gpu_index"

  # 定义日志文件名
  mkdir -p "log"
  log_file="log/${model}_${dataset}.log"

  # 启动新的进程
  mkdir -p "result"
  CUDA_VISIBLE_DEVICES=$gpu_index python eval/test_local_models.py \
    --model ${model} \
    --dataset ${dataset} \
    --test_setting ${test_setting} \
    --batch_size 32 \
    --max_new_tokens 10 > "$log_file" 2>&1 &

  # 记录进程的PID
  pids+=($!)
  echo "Pids of $model on $dataset: $!"

  # 等待一会再启动下一个进程
  sleep 30
done

# 等待所有进程完成
wait