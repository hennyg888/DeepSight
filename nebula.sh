#!/bin/bash
# LLaMA-Factory 多卡分布式训练脚本
# LLaMA-Factory multi-GPU distributed training script
#
# 默认用全部可见 GPU；如需指定，取消下面 CUDA_VISIBLE_DEVICES 那行的注释并修改
# By default uses all visible GPUs; to restrict, uncomment and edit CUDA_VISIBLE_DEVICES below
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 让 llamafactory-cli 自动用 torchrun 启动多卡训练
# Tell llamafactory-cli to launch via torchrun for multi-GPU training
export FORCE_TORCHRUN=1

# wandb 项目名（run 名在 configs/ad_bev_v4.yaml 的 run_name 里设）
# wandb project name (run name lives in configs/ad_bev_v4.yaml's run_name field)
export WANDB_PROJECT=DeepSight

# method1 (推荐 / preferred)
# 注意：YAML 必须作为位置参数，不能写 --config（CLI 通过 sys.argv[1].endswith('.yaml') 判断）
# NOTE: YAML must be a positional arg, NOT --config (CLI checks sys.argv[1].endswith('.yaml'))
llamafactory-cli train ./configs/ad_bev_v4.yaml

# method2 (备用 / fallback): 直接调用 torchrun
# nproc_per_node 改成你机器的 GPU 数
# Change nproc_per_node to match your GPU count
# torchrun --nproc_per_node=8 src/train.py ./configs/ad_bev_v4.yaml
