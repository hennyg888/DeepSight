import torch
from transformers import Qwen2_5_VLForConditionalGeneration as qwen25
from transformers import Qwen2_5_VLConfig as qwen25_config
from transformers import DINOv3ViTModel as dinov3
import time
from safetensors.torch import load_file


start_time = time.time()
# config_file = '/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen2.5-VL-3B-Instruct/config.json'
# config = qwen25_config.from_json_file(config_file)
# model = qwen25._from_config(config)
# model = qwen25.from_pretrained('/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen2.5-VL-3B-Instruct')
# 合并模型参数
# Merge model parameters
model = qwen25.from_pretrained('/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen2.5-VL-3B-Instruct')
print('模型初始化完成')
# dino_path = '/mnt/nas-data-1/wuchangjie.wcj/work/open_source/dinov3-vitL16/model.safetensors'
# message = model.dinov3.load_state_dict(load_file(dino_path))
# print(message)
model.to(torch.bfloat16)
model.save_pretrained('/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen2.5-VL-3B-Instruct-dino_v3-bf16')

# 测试模型参数
# Test model parameters
# print('load model qwen2.5-vl-3b-instruct')
# model = qwen25.from_pretrained('/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen2.5-VL-3B-Instruct-dino_v3')

# print('load model dinov3-vitL16')
# dino = dinov3.from_pretrained('/mnt/nas-data-1/wuchangjie.wcj/work/open_source/dinov3-vitL16')

# print('开始比较模型参数')
# state_dict1 = model.dinov3.state_dict()
# state_dict2 = dino.state_dict()

# # 检查键是否一致
# # Check if parameter keys match
# if state_dict1.keys() != state_dict2.keys():
#     print("参数键不一致")

# # 依次比较每个参数张量
# # Compare each parameter tensor one by one
# for key in state_dict1.keys():
#     tensor1 = state_dict1[key]
#     tensor2 = state_dict2[key]
#     # 使用 torch.allclose 比较是否所有元素都近似相等
#     # Use torch.allclose to check element-wise approximate equality
#     if not torch.allclose(tensor1, tensor2):
#         print(f"参数 {key} 不一致")

# print('模型参数比较完成')