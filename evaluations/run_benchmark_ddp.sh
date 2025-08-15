#/zhoujingbo/yk/work/OpenToMe/evaluations/run_benchmark_ddp.sh
#!/bin/bash

# --- 配置 ---
# 设置你想使用的 GPU 数量
NUM_GPUS=1

# 设置 PYTHONPATH，使其包含你的项目根目录
export PYTHONPATH=/yuchang/yk/work/OpenToMe:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
# 设置输出目录
OUTPUT_DIR="results_ddp"

echo "========================================================"
echo "Starting DDP Benchmark with ${NUM_GPUS} GPUs..."
echo "PYTHONPATH is set to: ${PYTHONPATH}"
echo "========================================================"

# --- 启动命令 ---
# 使用 torchrun 来启动分布式任务
# --standalone: 表示这是一个单机任务
# --nproc_per_node: 指定在本机上启动多少个进程 (即使用多少个 GPU)
# 后面跟着你要执行的常规 python 命令和参数

torchrun --standalone --nproc_per_node=${NUM_GPUS} evaluations/run_benchmark_ddp.py \
    --tome-variants none global_tome local_tome naive_local_tome \
    --source-tracking-modes matrix map \
    --local-h-ratio 0.05 \
    --model-names deit_small_patch16_224 \
    --seq-lens 32768\
    --target-ratios 0.5 \
    --batch-size 16 \
    --output-dir ${OUTPUT_DIR}