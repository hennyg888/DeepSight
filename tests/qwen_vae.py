import torch
from diffusers.models import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor
import os
from PIL import Image
# from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests

from typing import Any, Callable, Dict, List, Optional, Union


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_ori_imageimg.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def encode_vae_image(image: torch.Tensor, vae: AutoencoderKLQwenImage, generator: torch.Generator):

    image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode="argmax")
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    norm_image_latents = (image_latents - latents_mean) / latents_std

    return image_latents, norm_image_latents


def test_vae(vae):

    # image = Image.open(BytesIO(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg").content))
    image_path = 'debug/debug_2.jpg'
    image = Image.open(image_path)
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    latent_channels = vae.config.z_dim
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    height, width = 256, 512
    image = image_processor.resize(image, height, width)
    resized_image = image
    image = image_processor.preprocess(image, height, width)
    image = image.unsqueeze(2).to(torch.bfloat16).to('cuda')

    with torch.no_grad():
        image_latents, norm_image_latents = encode_vae_image(image, vae, None)
        print(image_latents.shape)

        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            image_latents.device, image_latents.dtype
        )
        norm_latents = norm_image_latents / latents_std + latents_mean
        norm_image = vae.decode(norm_latents, return_dict=False)[0][:, :, 0]
        norm_image = norm_image.to(torch.float32)
        norm_image = image_processor.postprocess(norm_image, output_type="pil")

        ori_image = vae.decode(image_latents, return_dict=False)[0][:, :, 0]
        ori_image = ori_image.to(torch.float32)
        ori_image = image_processor.postprocess(ori_image, output_type="pil")
        print(len(norm_image), len(ori_image))
        norm_image, ori_image = norm_image[0], ori_image[0]


        # norm_image.save("norm_image.png")
        # ori_image.save("ori_image.png")

        # 创建新图片，宽度为统一后的宽度，高度为三张图高度之和
        # Create a new canvas: unified width, height = sum of all image heights
        image = Image.open(image_path)
        total_height = image.height + resized_image.height + ori_image.height + norm_image.height
        new_img = Image.new('RGB', (resized_image.width, total_height))

        # 粘贴三张图片到新图上
        # Paste the images onto the new canvas
        new_img.paste(image, (0, 0))
        new_img.paste(resized_image, (0, image.height))  # 第一张图在顶部 / first image at top
        new_img.paste(ori_image, (0, image.height + resized_image.height))  # 第二张图在中间 / second image in middle
        new_img.paste(norm_image, (0, image.height + resized_image.height + ori_image.height))  # 第三张图在底部 / third image at bottom

        # 保存结果
        # Save the result
        new_img.save('vertical_concatenated_2.jpg')


if __name__ == '__main__':

    # vae_model_path = '/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen-Image-Edit/vae'
    # vae = AutoencoderKLQwenImage.from_pretrained(vae_model_path)
    # vae.to(torch.bfloat16)
    # vae.to("cuda")
    # vae.eval()

    # 
    from transformers import Qwen2_5_VLForConditionalGeneration as qwen25
    from transformers import Qwen2_5_VLConfig as qwen25_config
    import time
    start_time = time.time()
    # config_file = '/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen2.5-VL-3B-Instruct/config.json'
    # config = qwen25_config.from_json_file(config_file)
    # model = qwen25._from_config(config)
    # model = qwen25.from_pretrained('/mnt/nas-data-1/wuchangjie.wcj/work/open_source/Qwen2.5-VL-3B-Instruct')
    model = qwen25.from_pretrained('/mnt/nas-data-1/wuchangjie.wcj/work/debug/ex1/checkpoint-40')
    print('模型初始化完成')
    model.to(torch.bfloat16)
    model.to("cuda")
    end_time = time.time()
    print(f"load model time: {(end_time - start_time)/60} min")
    vae = model.model.vae
    vae.eval()
    
    test_vae(vae)