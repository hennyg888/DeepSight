#!/bin/bash
# LLaMA-Factory 8卡分布式训练脚本
# LLaMA-Factory 8-GPU distributed training script

# method1
llamafactory-cli train --config ./configs/ad_bev_v4.yaml

# method2
torchrun --nproc_per_node=8 src/train.py --config ./configs/ad_bev_v4.yaml
