# import os
# import json
# import random
# import numpy as np
# import cv2
# # from diffusers.image_processor import VaeImageProcessor
# import torch
# import torchvision.transforms.functional as TF
# from torchvision.transforms import v2
# from PIL import Image
# from ..extras.constants import IGNORE_INDEX


# IMAGE_SIZE = 256
# PATCH_SIZE = 16
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)


# def resize_and_normalize_transform(
#     image: Image,
#     image_size: int = IMAGE_SIZE,
#     patch_size: int = PATCH_SIZE,
# ) -> torch.Tensor:
#     image = image.convert('RGB')
#     w = 512
#     h = 512
#     crop_width, crop_height = 512, 512
#     left = (w - crop_width) // 2
#     top = 200
#     right = left + crop_width
#     bottom = top + crop_height
#     box = (left, top, right, bottom)
#     cropped_img = image.crop(box)
#     # croped_img = image[top:bottom, left:right]
#     h_patches = int(image_size / patch_size)
#     w_patches = int((w * image_size) / (h * patch_size))
#     image_resized = TF.to_tensor(TF.resize(cropped_img, (h_patches * patch_size, w_patches * patch_size)))
#     image_tensor = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
#     return image_tensor


# def make_transform(resize_size: int = 256):
#     to_tensor = v2.ToImage()
#     resize = v2.Resize((resize_size, resize_size), antialias=True)
#     to_float = v2.ToDtype(torch.float32, scale=True)
#     normalize = v2.Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#     )
#     return v2.Compose([to_tensor, resize, to_float, normalize])


# class ADCollector():

#     def __init__(self, token_processor, image_processor, tokenizer, img_size=(448, 896), augment=False, finetune=False):
#         self.token_processor = token_processor
#         self.image_processor = image_processor
#         self.tokenizer = tokenizer
#         # self.vae_image_processor = VaeImageProcessor()
#         self.img_size_h, self.img_size_w = img_size
#         self.augment = augment
#         self.patch_size = 16
#         self.time_step = 5

#     def label_processor(self, label=None):

#         caption = 'hello world'
#         assistant_caption = 'hello world'
#         command = int(label)

#         return caption, assistant_caption, command
    
#     # def image_argument(self, images, thr_sat=0.15, thr_trace=0.1, thr_sd=0.1):
#     def image_argument(self, images, thr_sat=0.3, thr_trace=0.3, thr_sd=0.3, thr_socol=0.4):
#         if random.random() < thr_sat:
#             if random.random() < 0.5:
#                 images[0] = self.empty_sat_image
#             else:
#                 random_key = self.image_keys[random.randint(0, self.num_key - 1)]
#                 images[0] = os.path.join(self.sat_path, f'{random_key}.jpg')
#         if random.random() < thr_trace:
#             if random.random() < 0.5:
#                 images[1] = self.empty_trace_image
#             else:
#                 random_key = self.image_keys[random.randint(0, self.num_key - 1)]
#                 images[1] = os.path.join(self.trace_path, f'{random_key}.jpg')
#         if random.random() < thr_sd:
#             images[2] = self.empty_sat_image
            
#         # # 有一定概率, 只有socol.
#         # if random.random() < thr_socol:
#         #     images[0] = self.empty_sat_image
#         #     images[1] = self.empty_trace_image
#         #     images[2] = self.empty_sat_image

#         # if random.random() < thr_trace_low:
#         #     if random.random() < 0.5:
#         #         images[3] = self.empty_trace_image
#         #     else:
#         #         random_key = self.image_keys[random.randint(0, self.num_key - 1)]
#         #         images[3] = os.path.join(self.trace_low_path, f'{random_key}.jpg')
#         return images


#     def __call__(self, samples):
#         # from pudb import set_trace; set_trace()
#         batch_samples = {}
#         batch_size = len(samples)
#         batch_target_tensors = []
#         for i in range(batch_size):
#             # 设置默认值
#             samples[i]['_videos'] = None
#             samples[i]['_audios'] = None  
#             target_images = [] 
#             for _ in range(self.time_step):
#                 # 最后五张为 BEV 图像
#                 target_img_file = samples[i]['_images'].pop()
#                 target_img = Image.open(target_img_file)
#                 target_img_tensor = resize_and_normalize_transform(target_img)
#                 target_images.insert(0, target_img_tensor)
#             batch_target_tensors.append(torch.stack(target_images, dim=0)) # (5, 3, 256, 256)
#         target_tensors = torch.stack(batch_target_tensors, dim=0) #(B, 5, 3, 256, 256)

#         # 处理文本token
#         keys = samples[0].keys()
#         batch_samples = {k:[samples[i][k] for i in range(batch_size)] for k in keys}
#         batch_samples = self.token_processor(batch_samples)
        
#         # 处理 bev 相关的label
#         start_bev_token_id = self.tokenizer.convert_tokens_to_ids('<|start_bev_token|>')
#         end_bev_token_id = self.tokenizer.convert_tokens_to_ids('<|end_bev_token|>')
#         b, t, c, h, w = target_tensors.shape
#         template_mask = torch.ones((t, h * w // self.patch_size ** 2 + 5), dtype=torch.bool)
#         template_mask[:, 1:5] = False
#         template_mask = template_mask.reshape(-1)
#         keys = batch_samples.keys()
#         samples = [{k:batch_samples[k][i] for k in keys} for i in range(batch_size)]
#         batch_label_bev_masks = []
#         for bi, sample in enumerate(samples):
#             # input_ids = sample['input_ids']
#             labels = sample['labels']
#             # from pudb import set_trace; set_trace()
#             start_index = labels.index(start_bev_token_id)
#             end_index = labels.index(end_bev_token_id)
#             labels[start_index + 1:end_index] = [IGNORE_INDEX] * (end_index - start_index - 1)
#             label_bev_masks = torch.zeros((len(labels),), dtype=torch.bool)
#             label_bev_masks[start_index + 1:end_index] = template_mask
#             batch_label_bev_masks.append(label_bev_masks)
#             assert end_index - start_index - 1 == (h * w // self.patch_size ** 2 + 5) * t
        
#         # 处理pv图像token for qwen_vit
#         samples = self.image_processor(samples)
        
#         # 处理bev图像token for vae
        
        
#         samples['pixel_values_bevs'] = target_tensors.to(samples["pixel_values"].dtype)
#         samples['bevs_masks'] = template_mask.unsqueeze(0).repeat(b, 1)  #(b, l_v)
#         # padding label_bev_masks
#         b, l = samples['labels'].shape
#         padding_label_bev_masks = torch.zeros((b, l), dtype=torch.bool)
#         for bi, label_bev_masks in enumerate(batch_label_bev_masks):
#             padding_label_bev_masks[bi, :label_bev_masks.shape[0]] = label_bev_masks
#         samples['label_bev_masks'] = padding_label_bev_masks  #(b, l)

#         # 处理时间步
#         # time_step = torch.tensor([[0, 0.25, 0.5, 0.75, 1.0]] * batch_size)
#         # samples['timestep'] = time_step.permute(1, 0).to(samples["pixel_values"].dtype)
#         # samples['commands'] = torch.tensor(batch_command)

#         return samples


# if __name__ == '__main__':

#     pass
import os
import json
import random
import numpy as np
import cv2
# from diffusers.image_processor import VaeImageProcessor
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
from PIL import Image
from ..extras.constants import IGNORE_INDEX


IMAGE_SIZE = 256
PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_and_normalize_transform(
    image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    image = image.convert('RGB')
    w, h = IMAGE_SIZE, IMAGE_SIZE
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    image_resized = TF.to_tensor(TF.resize(image, (h_patches * patch_size, w_patches * patch_size)))
    image_tensor = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return image_tensor


def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class ADCollector():

    def __init__(self, token_processor, image_processor, tokenizer, img_size=(448, 896), augment=False, finetune=False):
        self.token_processor = token_processor
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        # self.vae_image_processor = VaeImageProcessor()
        self.img_size_h, self.img_size_w = img_size
        self.augment = augment
        self.patch_size = 16
        self.time_step = 5

    def label_processor(self, label=None):

        caption = 'hello world'
        assistant_caption = 'hello world'
        command = int(label)

        return caption, assistant_caption, command
    
    # def image_argument(self, images, thr_sat=0.15, thr_trace=0.1, thr_sd=0.1):
    def image_argument(self, images, thr_sat=0.3, thr_trace=0.3, thr_sd=0.3, thr_socol=0.4):
        if random.random() < thr_sat:
            if random.random() < 0.5:
                images[0] = self.empty_sat_image
            else:
                random_key = self.image_keys[random.randint(0, self.num_key - 1)]
                images[0] = os.path.join(self.sat_path, f'{random_key}.jpg')
        if random.random() < thr_trace:
            if random.random() < 0.5:
                images[1] = self.empty_trace_image
            else:
                random_key = self.image_keys[random.randint(0, self.num_key - 1)]
                images[1] = os.path.join(self.trace_path, f'{random_key}.jpg')
        if random.random() < thr_sd:
            images[2] = self.empty_sat_image
            
        # # 有一定概率, 只有socol.
        # if random.random() < thr_socol:
        #     images[0] = self.empty_sat_image
        #     images[1] = self.empty_trace_image
        #     images[2] = self.empty_sat_image

        # if random.random() < thr_trace_low:
        #     if random.random() < 0.5:
        #         images[3] = self.empty_trace_image
        #     else:
        #         random_key = self.image_keys[random.randint(0, self.num_key - 1)]
        #         images[3] = os.path.join(self.trace_low_path, f'{random_key}.jpg')
        return images


    def __call__(self, samples):
        # from pudb import set_trace; set_trace()
        batch_samples = {}
        batch_size = len(samples)
        batch_target_tensors = []
        for i in range(batch_size):
            # 设置默认值
            # Set default values
            samples[i]['_videos'] = None
            samples[i]['_audios'] = None  
            target_images = [] 
            for _ in range(self.time_step):
                # 最后五张为 BEV 图像
                # The last five images are BEV images
                target_img_file = samples[i]['_images'].pop()
                target_img = Image.open(target_img_file)
                target_img_tensor = resize_and_normalize_transform(target_img)
                target_images.insert(0, target_img_tensor)
            batch_target_tensors.append(torch.stack(target_images, dim=0)) # (5, 3, 256, 256)
        target_tensors = torch.stack(batch_target_tensors, dim=0) #(B, 5, 3, 256, 256)

        # 处理文本token
        # Process text tokens
        keys = samples[0].keys()
        batch_samples = {k:[samples[i][k] for i in range(batch_size)] for k in keys}
        batch_samples = self.token_processor(batch_samples)
        
        # 处理 bev 相关的label
        # Process BEV-related labels
        start_bev_token_id = self.tokenizer.convert_tokens_to_ids('<|start_bev_token|>')
        end_bev_token_id = self.tokenizer.convert_tokens_to_ids('<|end_bev_token|>')
        b, t, c, h, w = target_tensors.shape
        template_mask = torch.ones((t, h * w // self.patch_size ** 2 + 5), dtype=torch.bool)
        template_mask[:, 1:5] = False
        template_mask = template_mask.reshape(-1)
        keys = batch_samples.keys()
        samples = [{k:batch_samples[k][i] for k in keys} for i in range(batch_size)]
        batch_label_bev_masks = []
        for bi, sample in enumerate(samples):
            # input_ids = sample['input_ids']
            labels = sample['labels']
            # from pudb import set_trace; set_trace()
            start_index = labels.index(start_bev_token_id)
            end_index = labels.index(end_bev_token_id)
            labels[start_index + 1:end_index] = [IGNORE_INDEX] * (end_index - start_index - 1)
            label_bev_masks = torch.zeros((len(labels),), dtype=torch.bool)
            label_bev_masks[start_index + 1:end_index] = template_mask
            batch_label_bev_masks.append(label_bev_masks)
            assert end_index - start_index - 1 == (h * w // self.patch_size ** 2 + 5) * t
        
        # 处理pv图像token for qwen_vit
        # Process perspective-view image tokens for Qwen ViT
        samples = self.image_processor(samples)

        # 处理bev图像token for vae
        # Process BEV image tokens for VAE
        
        
        samples['pixel_values_bevs'] = target_tensors.to(samples["pixel_values"].dtype)
        samples['bevs_masks'] = template_mask.unsqueeze(0).repeat(b, 1)  #(b, l_v)
        # padding label_bev_masks
        b, l = samples['labels'].shape
        padding_label_bev_masks = torch.zeros((b, l), dtype=torch.bool)
        for bi, label_bev_masks in enumerate(batch_label_bev_masks):
            padding_label_bev_masks[bi, :label_bev_masks.shape[0]] = label_bev_masks
        samples['label_bev_masks'] = padding_label_bev_masks  #(b, l)

        # 处理时间步
        # Process timestep inputs
        # time_step = torch.tensor([[0, 0.25, 0.5, 0.75, 1.0]] * batch_size)
        # samples['timestep'] = time_step.permute(1, 0).to(samples["pixel_values"].dtype)
        # samples['commands'] = torch.tensor(batch_command)

        return samples


if __name__ == '__main__':

    pass