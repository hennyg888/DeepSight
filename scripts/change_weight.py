from safetensors.torch import load_file, save_file
from safetensors import safe_open
import torch


input_weight_path_1 = 'model-00001-of-00002.safetensors'
input_weight_path_2 = 'model-00002-of-00002.safetensors'
output_weight_path = 'model.safetensors'


with safe_open(input_weight_path_1, framework="pt") as f:
    metadata = f.metadata()
model1_state_dict = load_file(input_weight_path_1)
model2_state_dict = load_file(input_weight_path_2)
# 合并模型
# Merge model shards
merged_model = {**model1_state_dict, **model2_state_dict}
save_file(merged_model, "model.safetensors", metadata)
print("done")


# 更改t的维度
# Modify the temporal dimension of the patch embedding weight
file_path = 'model.safetensors'
state_dict = load_file(file_path)
with safe_open(file_path, framework="pt") as f:
    metadata = f.metadata()
w = state_dict['visual.patch_embed.proj.weight']
w = torch.concat([w,w[:, :, :1, :, :]], dim=2)
state_dict['visual.patch_embed.proj.weight'] = w

# 增加 vae 模块
# Add the VAE module weights

vae_file = '/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors'
vae_state_dict = load_file(vae_file)
for k, v in vae_state_dict.items():
    state_dict['model.vae.'+k] = v

save_file(state_dict, file_path, metadata)