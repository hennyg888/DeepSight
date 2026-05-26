import os
import json
import random
import numpy as np
import cv2
# from diffusers.image_processor import VaeImageProcessor
from collections import defaultdict
import math
from shapely.geometry import Polygon
import copy
import torch
from PIL import Image
# Stub：本 repo 中缺失可视化 / label 工具模块。
# Stub: visualization / label-utils modules are missing from this repo.
# RoadCollector 仅被 export，训练路径里实际用的是 ADCollector（workflow.py 选用）。
# RoadCollector is only exported; the training path actually uses ADCollector (workflow.py picks it).
# 这里写 stub 让 module 能正常 import；若有代码真调到这些函数，会立刻抛 NotImplementedError 而不是悄悄出错。
# Stubs let the module import cleanly; if anything actually calls these, NotImplementedError fires loud rather than silently misbehaving.
def _missing_util(*args, **kwargs):
    raise NotImplementedError(
        "RoadCollector visualization/label utils are not bundled in this repo. "
        "If you hit this, you probably wanted ADCollector instead."
    )
visual_objs = _missing_util
visual_road = _missing_util
get_sub_type_color_2 = _missing_util
visual_line_with_arrow = _missing_util
merge_classes_and_ranges = _missing_util
get_range_point = _missing_util


def draw_arrow(image, start_point, angle, length=15, arrow_color=(0, 255, 0), thickness=2):
    # 线段长度
    # Line length

    # 将角度转换为弧度
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    # 计算终点坐标
    # Compute the endpoint coordinates
    # 注意：在图像坐标系中，y轴是向下的，因此角度需要调整
    # Note: in image coordinates the y-axis points downward, so the angle must be adjusted
    delta_x = length * np.sin(angle_rad)
    delta_y = -length * np.cos(angle_rad)  # 负号是因为y轴向下 / negated because the y-axis points downward
    end_point = (int(start_point[0] + delta_x), int(start_point[1] + delta_y))
    start_point = (int(start_point[0]), int(start_point[1]))
    # 绘制带箭头的线段
    # Draw the arrowed line segment
    cv2.arrowedLine(image, start_point, end_point, arrow_color, thickness, tipLength=0.5)
    return image


def calculate_perimeter(points):
    """
    计算坐标列表的周长。
    Compute the perimeter of a list of coordinates.

    参数:
    Args:
    points (list): 坐标列表，每个元素是一个 (x, y) 元组。
        points (list): List of (x, y) coordinate tuples.

    返回:
    Returns:
    float: 周长。
        float: Perimeter length.
    """
    perimeter = 0.0
    num_points = len(points)
    for i in range(num_points-1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]  # 循环到第一个点 / advance to the next point
        perimeter += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return perimeter


def sample_points_v2(socol_images, socol_points, socol_angles, samples_num):
    """
    根据周长采样图像、坐标和角度列表。
    Sample images, coordinates, and angles according to the perimeter.

    参数:
    Args:
    socol_images (list): 图像列表。
        socol_images (list): List of images.
    socol_points (list): 坐标列表，每个元素是一个 (x, y) 元组。
        socol_points (list): List of (x, y) coordinate tuples.
    socol_angles (list): 角度列表。
        socol_angles (list): List of angles.

    返回:
    Returns:
    tuple: 采样得到的图像、坐标和角度列表。
        tuple: Sampled images, coordinates, and angles.
    """
    if not socol_images or not socol_points or not socol_angles:
        return [], [], []

    # 计算周长
    # Compute the perimeter
    perimeter = calculate_perimeter(socol_points)

    # 根据周长决定采样策略
    # Choose the sampling strategy based on perimeter
    if perimeter < 500:
        num_samples = samples_num  # 等间距采样 6 个样本 / evenly spaced, 6 samples
    else:
        # 以间距 90 采样，最多 10 个样本
        # Sample at ~90-unit intervals, capped at 10 samples
        num_samples = min(10, int(perimeter // 90) + 1)

    # 确保采样点数不超过列表长度
    # Ensure the sample count does not exceed the list length
    num_samples = min(num_samples, len(socol_images))

    # 计算步长
    # Compute the step size
    step = (len(socol_images)) / (num_samples) if num_samples > 1 else 0

    # 采样点
    # Sample indices
    sampled_indices = [int(round(i * step)) for i in range(num_samples)]
    sampled_socol_images = [socol_images[idx] for idx in sampled_indices]
    sampled_socol_points = [socol_points[idx] for idx in sampled_indices]
    sampled_socol_angles = [socol_angles[idx] for idx in sampled_indices]

    return sampled_socol_images, sampled_socol_points, sampled_socol_angles


def mean_angle(angles):
    """计算一组角度的均值 / Compute the mean of a set of angles."""
    sum_sin = np.sum(np.sin(angles))
    sum_cos = np.sum(np.cos(angles))
    mean_angle = np.arctan2(sum_sin, sum_cos)
    return mean_angle


def get_main_direction(line):
    """计算线段的主方向 / Compute the principal direction of a polyline."""
    # 从线段的点集中提取每个小段的起始和结束点
    # Extract start and end points of each segment from the point set
    segments = zip(line[:-1], line[1:])

    angles = []
    for (x1, y1), (x2, y2) in segments:
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)
        angles.append(angle)

    # 计算所有小段角度的均值，作为主方向
    # Average all segment angles to obtain the principal direction
    main_direction = mean_angle(np.array(angles))

    return main_direction


def convert_label_to_line(label):
    line = [label["start_point"]] + label["sample_points"] + [label["end_point"]]
    return line

def disturb_objs_x(hq_objs, augument):
    disturb_range = 20
    for obj in hq_objs:
        obj_types = obj['obj_type']
        obj_coords = obj['coord']
        if augument:
            if random.random() < 0.25:
                for obj_center in obj_coords:
                    obj_center[0] += int(disturb_range * (random.random() - 0.5))
                    obj_center[1] += int(disturb_range * (random.random() - 0.5))
                if int(obj_types) in [1, 600]:
                    obj_coords[0] = obj_coords[-1]
    hq_objs_add = []
    if augument:
        for obj in hq_objs:
            if random.random() < 0.5:
                hq_objs_add.append(obj)
        if random.random() < 0.75:  # 不加ref / do not add ref
            hq_objs_add = []
    else:
        hq_objs_add = hq_objs
    return hq_objs_add

class RoadCollector():

    def __init__(self, token_processor, image_processor, img_size=(448, 896), augment=False, finetune=False):
        self.token_processor = token_processor
        self.image_processor = image_processor
        self.vae_image_processor = VaeImageProcessor()
        self.img_size_h, self.img_size_w = img_size
        self.augment = augment
        self.finetune = finetune
        self.data_root = '/mnt/nas-data-1/slz/data/roadgpt/train_qwen_v34_v1' 
        if self.augment:
            with open(f'{self.data_root}/keys.json') as fp:
                self.image_keys = json.load(fp)
        else:
            self.image_keys = []
        self.num_key = len(self.image_keys)
        self.sat_path = f'{self.data_root}/images'
        self.trace_path = f'{self.data_root}/traces'
        # self.trace_low_path = f'{self.data_root}/traces_filter'
        self.visual_sd_path = os.path.abspath(os.path.dirname(__file__)) + "/../../../../data/visual_sd"
        self.target_road_path = os.path.abspath(os.path.dirname(__file__)) + "/../../../../data/visual_hq"
        os.makedirs(self.visual_sd_path, exist_ok=True)
        os.makedirs(self.target_road_path, exist_ok=True)
        self.empty_sat_image = os.path.abspath(os.path.dirname(__file__)) + "/../../../data/empty_img.jpg"
        self.empty_trace_image = os.path.abspath(os.path.dirname(__file__)) + "/../../../data/empty_trace.jpg"

    def label_processor(self, label, visual_prompt=None, sd_vis_path=None, use_gt=False, use_class=False):
        """
        label示例：
        Example label:
            /mnt/gt-data2/wuxiang/projects/LLaMA-Factory/output/road_collector_label_inputs_eg.txt
            /mnt/gt-data2/wuxiang/projects/LLaMA-Factory/output/infer_road_collector_label_inputs_eg.txt

        输出caption, assistant_caption示例
        Example output caption and assistant_caption:
        caption：
            /mnt/gt-data2/wuxiang/projects/LLaMA-Factory/output/road_collector_label_user_content_eg.txt
            /mnt/gt-data2/wuxiang/projects/LLaMA-Factory/output/infer_road_collector_label_user_content_eg.txt
        assistant_caption：
            /mnt/gt-data2/wuxiang/projects/LLaMA-Factory/output/road_collector_label_assistant_content_eg.txt
            /mnt/gt-data2/wuxiang/projects/LLaMA-Factory/output/infer_road_collector_label_assistant_content_eg.txt
        """
        # 做visual prompt用
        # Used for constructing visual prompts
        pred_sd = []
        ref_hq = []

        label = json.loads(label)

        ref_sd_lines = label["total_crop_sds"]
        ref_sd_z_levels = label["total_crop_sd_z_levels"]
        num_sd = len(ref_sd_lines)
        sd_indexs = list(range(num_sd))
        if self.augment and num_sd > 1:
            random.shuffle(sd_indexs)
        cur_z_level_sd_indexs = set()
        not_cur_z_level_sd_indexs = set()
        for sd_key, sd_index in enumerate(sd_indexs):
            if ref_sd_lines[sd_index].get("is_current_z_level", True):
                cur_z_level_sd_indexs.add(sd_index)
            else:
                not_cur_z_level_sd_indexs.add(sd_index)
        # if len(not_cur_z_level_sd_indexs) > 0:
        #     print(f"cur_z_level_sd_indexs: {len(cur_z_level_sd_indexs)}, not_cur_z_level_sd_indexs: {len(not_cur_z_level_sd_indexs)}, sd_vis_path: {sd_vis_path}")

        ref_sd_captions = []
        for sd_key, sd_index in enumerate(sd_indexs):
            if ((not use_gt) and sd_index not in cur_z_level_sd_indexs):
                continue
            ref_sd_line = ref_sd_lines[sd_index]
            line_caption = self.line2caption(ref_sd_line, 'sd_line', mask_flag=False)
            line_caption = f'|sd_line|:{line_caption}'
            z_level_caption = f'|sd_z_level|:{ref_sd_z_levels[sd_index]}'
            line_caption = f'|sd_{sd_key+1}|:' + '{' + line_caption + ',' + z_level_caption + '}'
            # line_caption = f'|sd_{sd_key+1}|:' + '{' + line_caption + '}'
            ref_sd_captions.append(line_caption)
        mask_flag = self.augment and (random.random() < 0.10)
        line_caption = self.line2caption(label['sd_line'], 'sd_line', mask_flag=mask_flag)
        line_caption = f'|sd_line|:{line_caption}'
        z_level_caption = f'|sd_z_level|:{label["sd_z_level"]}'
        line_caption = f'|sd_{num_sd+1}|:' + '{' + line_caption + ',' + z_level_caption + '}'
        # line_caption = f'|sd_{num_sd+1}|:' + '{' + line_caption + '}'
        ref_sd_captions.append(line_caption)
        ref_sd_captions = f'[{",".join(ref_sd_captions)}]'

        ref_hq_captions = []
        pred_hq_captions = []
        pred_sd_key_and_ref_points = []
        total_hq_road = label['total_crop_hqs']
        total_crop_objs_x = label['total_crop_objs_x']
        for sd_key, sd_index in enumerate(sd_indexs):
            hq_road = total_hq_road[sd_index]
            hq_objs = total_crop_objs_x[sd_index]
            if len(hq_road) == 0 or len(hq_road) >= 5 or ((not use_gt) and sd_index not in cur_z_level_sd_indexs): # NOTE 这里其他层的hq也放进来，原因是context里的sd没有车道数信息 / other-layer HQ is included here because the SD in context lacks lane-count info
                continue
            hq_objs_caption = self.obj2caption(hq_objs)
            road_caption, ref_caption = self.road2caption(hq_road, hq_objs_caption, sd_key,
                                                          use_class=use_class)
            if self.augment:
                hq_objs = disturb_objs_x(hq_objs, self.augment)
                hq_objs_caption_disturb = self.obj2caption(hq_objs)
                road_caption_disturb, _ = self.road2caption(hq_road, hq_objs_caption_disturb, sd_key,
                                                                      use_class=use_class)
                total_crop_objs_x[sd_index] = copy.deepcopy(hq_objs)

                if random.random() > 0.5:     # 用来当作ref / used as a reference sample
                    ref_hq_captions.append(road_caption_disturb)
                    ref_hq.append(sd_index)
                elif random.random() > 0.5:  # 用来当作预测 / used as a prediction target
                    pred_flag = label['total_crop_pred_flags'][sd_index]
                    if pred_flag:
                        pred_sd_key_and_ref_points.append(ref_caption)
                        pred_hq_captions.append(road_caption)
                        # 用于预测:
                        # Used for prediction:
                        pred_sd.append(sd_index)
                else:
                    pass
            else:
                ref_hq_captions.append(road_caption)
                ref_hq.append(sd_index)
        if self.finetune or (self.augment and random.random() < 0.25):
            pred_hq_captions = []
            pred_sd_key_and_ref_points = []
            # 用于预测:
            # Used for prediction:
            pred_sd = []

        hq_objs = total_crop_objs_x[num_sd]
        if self.augment:
            hq_objs = disturb_objs_x(hq_objs, self.augment)
            total_crop_objs_x[num_sd] = hq_objs
        ref_hq.append(num_sd)
        obj_caption = self.roadobj2caption(hq_objs, num_sd)
        ref_hq_captions.append(obj_caption)

        # 增加高铁输入 TODO
        # Add high-speed-rail input (TODO)
        use_obj = True
        ref_gt_caption = "{}"
        if use_gt and len(label.get('total_crop_gts', [])) > 0:
            total_gt_road = label['total_crop_gts']
            line_type2lines = defaultdict(list)
            obj_type2lines = defaultdict(list)
            # 计算z的范围
            # Compute the z-value range
            zs = []
            for road_idx, road in enumerate(total_gt_road):
                for line_type, lines in road.items():
                    for line in lines:
                        zs += [line["start_point"][2], line["end_point"][2], *[i[2] for i in line["sample_points"]]]
            # 把z转为层级
            # Convert z values to layer indices
            min_z = min(zs) if len(zs) > 0 else 0
            layer_z_meters, max_layers = 1, 100
            for road_idx, road in enumerate(total_gt_road):
                for line_type, lines in road.items():
                    for line in lines:
                        line["start_point"][2] = min((line["start_point"][2] - min_z) // layer_z_meters, max_layers)
                        line["end_point"][2] = min((line["end_point"][2] - min_z) // layer_z_meters, max_layers)
                        for i in line["sample_points"]:
                            i[2] = min((i[2] - min_z) // layer_z_meters, max_layers)
            # 获得caption
            # Build the caption strings
            for road_idx, road in enumerate(total_gt_road):
                for line_type, lines in road.items():
                    line_type = self.get_gt_line_type(line_type)
                    if line_type is None:
                        continue
                    if self.is_gt_obj(line_type) and use_obj:
                        objs = self.get_struct_objs(line_type, lines)
                        for obj in objs:
                            if 'prompt_points' not in obj: #TODO(ztd) polygon在crop patch后只剩一条线，暂时continue
                                continue
                            obj_caption = self.gtobj2caption(obj, line_type, mask_flag=False, easy_mode=True)
                            obj_type2lines[f"gt_obj_{line_type}"].append(obj_caption)
                        continue
                    for line in lines:
                        line_caption = self.line2caption(line, line_type, mask_flag=False, easy_mode=True)
                        line_type2lines[f"gt_line_{line_type}"].append(line_caption)
            line_captions = []
            for line_type, lines in line_type2lines.items():
                line_captions.append(f'|{line_type}|:[{";".join(lines)}]')
            if use_obj:
                for obj_type, objs in obj_type2lines.items():
                    line_captions.append(f'|{obj_type}|:[{";".join(objs)}]')
            ref_gt_caption = "{" + ",".join(line_captions) + "}" 
            # 以一定概率丢弃所有车道线
            # Randomly drop all lane lines with a small probability
            if (self.augment and random.random() < 0.1) or (self.finetune and len(ref_gt_caption) > 30000): # 4w ~=12000token
                ref_gt_caption = "{}"

        # "sd_infos": {"sd_road_num": sd_road_num, "cross_road_num": cross_road_num, "form_way": form_way, "sd_type": sd_type},
        sd_infos = label.get("sd_infos", None)
        sd_road_num, cross_road_num, form_way = None, None, None
        
        if sd_infos is not None:
            sd_road_num = sd_infos["sd_road_num"]
            sd_arrows = sd_infos["sd_road_arraws"] if "sd_road_arraws" in sd_infos else sd_infos["sd_road_arrows"]
            cross_road_num = sd_infos["cross_road_num"]
            form_way = sd_infos["form_way"]
            pre_length = sd_infos["pre_length"]
            suc_length = sd_infos["suc_length"]

            sd_furniture_bus_info = sd_infos.get('sd_furniture_bus_info', None)
            sd_lanedesc_bus_info = sd_infos.get('sd_lanedesc_bus_info', None)
            # ================================================
            # socol_nums = sd_infos["socol_road_nums"]
            # socol_nums = self.select_socol_road_nums(socol_nums)
            socol_nums = None
            # ================================================

        # import ipdb; ipdb.set_trace()
        hq_objs_caption = self.obj2caption(label.get('hd_objs_x', []))
        road_caption, ref_caption = self.road2caption(label['hd_roads'],
                                                      hq_objs_caption,
                                                      num_sd, 
                                                      sd_road_num=sd_road_num, 
                                                      sd_arrows=sd_arrows,
                                                      cross_road_num=cross_road_num, 
                                                      form_way=form_way,
                                                      pre_length=pre_length,
                                                      suc_length=suc_length,
                                                      sd_furniture_bus_info=sd_furniture_bus_info,
                                                      sd_lanedesc_bus_info=sd_lanedesc_bus_info,
                                                      socol_nums=socol_nums,
                                                      start_ref_points=label['start_ref_points'],
                                                      neighbor_ref_points=label['neighbor_ref_points'],
                                                      special_start_ref_points=label['special_start_ref_points'],
                                                      start_ref_sub_types=label.get('start_ref_sub_types', []),
                                                      start_ref_lane_types=label.get('start_ref_lane_types', []),
                                                      use_class=use_class)
        if len(label['end_ref_points']) > 0:
            if (not self.augment) or self.augment > 0.25:
                sd_key = f'|sd_{num_sd+1}|'
                topo_prompt = f"请先预测{sd_key}的topo点，然后"
                topo_points = [self.format_point(point) for point in label['end_ref_points']]
                topo_caption = '{' + f"{sd_key}:" + '{' + f'|topo_points|:[{",".join(topo_points)}]' + '}' + '}'
        pred_hq_captions.append(road_caption)
        pred_sd_key_and_ref_points.append(ref_caption)
        # 预测顺序调整为随机
        # Randomize the prediction order
        assert len(pred_hq_captions) == len(pred_sd_key_and_ref_points)
        num_pred = len(pred_hq_captions)
        shuffle_index = list(range(num_pred))
        random.shuffle(shuffle_index) # TODO 看上去是对需要预测的每个sd（可能预测多个sd）顺序进行随机 / randomize the order of each SD to be predicted (may predict multiple SDs)
        pred_hq_captions = [pred_hq_captions[i] for i in shuffle_index]
        pred_sd_key_and_ref_points = [pred_sd_key_and_ref_points[i] for i in shuffle_index]

        ref_hq_caption = f'[{",".join(ref_hq_captions)}]'
        pred_hq_caption = f'[{",".join(pred_hq_captions)}]'
        pred_sd_key_and_ref_points_captions = f'[{",".join(pred_sd_key_and_ref_points)}]'

        if self.augment:
            # caption ="<image>\n"
            caption = '<image>。'
            # socol_prompt, socol_images = self.get_socol_prompt_and_images(label)
            # socol_sample_points = label['socol_points']
            # socol_sample_angles = label['socol_images_angle']       
            # for socol_point, socol_angle in zip(socol_sample_points, socol_sample_angles):
            #     caption += f'{str(socol_point)}，{str(socol_angle)}，<video>。'
            caption += '\n'
        else:
            caption = "\n"

        caption = caption + ("请参考SD引导线和部分HQ数据，来补充需要制作的HQ数据。" if not use_gt else "请参考SD引导线和部分HQ数据，以及部分可能存在质量问题的模拟HQ数据，来补充需要制作的HQ数据。")
        caption = caption + f'SD引导线为：{ref_sd_captions}。'
        # caption = caption + f'部分HQ数据为：{ref_hq_caption}。'
        caption = caption + f'部分HQ数据已经绘制在图片上。'
        if use_gt:
            caption = caption + f'部分模拟HQ数据为：{ref_gt_caption}。'
        if len(label['end_ref_points']) > 0:
            caption = caption + topo_prompt
        caption = caption + f'请根据以下SD和参考点，预测HQ数据：{pred_sd_key_and_ref_points_captions}。'

        if len(label['end_ref_points']) > 0:
            assistant_caption = 'topo点为：' + topo_caption
        else: 
            assistant_caption = ''
        assistant_caption = assistant_caption + 'HQ数据为：' +pred_hq_caption
        # https://code.alibaba-inc.com/wuchangjie.wcj/LLaMA-Factory/blob/xinyuan_checkout_1216/src/llamafactory/data/road_collator.py#L163
        # 加入 Visual Prompt
        # Add the visual prompt
        # ===========================================================================================
        # 造图
        # Generate the image
        if self.augment and random.random() < 0.15: # TODO 高铁首次实验，去掉轨迹 / first high-speed-rail experiment: remove trajectory
            imgh, imgw = 448, 896
            white_img = np.zeros((imgh, imgw, 3), dtype=np.uint8)
        else:
            white_img = cv2.imread(visual_prompt)
        assert white_img is not None, f"图像读取失败: {visual_prompt}"
        target_img = np.zeros((256, 512, 3), dtype=np.uint8) + 255
        # setp1.画hq
        # Step 1. Draw HQ
        # context hq 信息
        # Context HQ information
        for _, sd_index in enumerate(sd_indexs):
            # 只能画被当做 ref 的 hq
            # Only draw HQ roads that are treated as references
            if (sd_index not in ref_hq) or (sd_index in not_cur_z_level_sd_indexs):
                continue
            hq_road = total_hq_road[sd_index]
            for road in hq_road:
                visual_road(white_img, road, use_class=use_class, color_map=get_sub_type_color_2)
                visual_road(target_img, road, use_class=use_class, color_map=get_sub_type_color_2, resize_tatio=8/14)
        for _, sd_index in enumerate(sd_indexs+[num_sd]):
            if (sd_index not in ref_hq) or (sd_index in not_cur_z_level_sd_indexs):
                continue
            hq_objs_x = total_crop_objs_x[sd_index]
            visual_objs(hq_objs_x, white_img)
        # 绘制 current hq
        # Draw the current-frame HQ road
        for road in label['hd_roads']:
            visual_road(target_img, road, use_class=use_class, color_map=get_sub_type_color_2, resize_tatio=8/14)

        assert white_img is not None, "绘制hq后图像为空"
        # step2.画SD
        # Step 2. Draw SD guide lines
        for _, sd_index in enumerate(sd_indexs):
            if sd_index in not_cur_z_level_sd_indexs:
                continue
            if sd_index not in pred_sd:     # 要预测 / this SD is to be predicted
                color = (0, 0, 255)
                thickness = 2
            else:
                color = (0, 255, 255)
                thickness = 4
            ref_sd_line = ref_sd_lines[sd_index]
            line = np.array(convert_label_to_line(ref_sd_line), dtype=np.int32)
            visual_line_with_arrow(line, white_img, color, thickness=thickness)
        line = np.array(convert_label_to_line(label['sd_line']), dtype=np.int32)
        color = (0, 255, 255)
        thickness = 4
        visual_line_with_arrow(line, white_img, color, thickness=thickness)
        assert white_img is not None, "绘制sd后图像为空"

        # step3. 画socol轨迹
        socol_sample_points = label['socol_points']
        socol_sample_angles = label['socol_images_angle']
        # 绘制轨迹和朝向
        # Draw trajectory points and heading arrows
        for point, angle in zip(socol_sample_points, socol_sample_angles):
            point = [int(point[0]), int(point[1])]
            cv2.circle(white_img, point, 4, (255, 130, 71), 1, cv2.LINE_AA)
            white_img = draw_arrow(white_img, point, angle)

        # step4 存图
        if sd_vis_path is not None:
            cv2.imwrite(sd_vis_path, white_img)
            target_vis_path = os.path.join(self.target_road_path, os.path.basename(sd_vis_path))
            cv2.imwrite(target_vis_path, target_img)
        # ===========================================================================================

        return caption, assistant_caption

    def select_socol_road_nums(self, sequence, num_samples=5):
        # 如果序列长度小于等于num_samples，直接返回整个序列
        # If the sequence is shorter than num_samples, return it as-is
        if len(sequence) <= num_samples:
            return sequence

        # 计算采样的步长
        # Compute the sampling step size
        step = len(sequence) / num_samples

        # 生成采样索引，并加入随机扰动
        # Generate sample indices with optional random jitter
        indices = []
        for i in range(num_samples):
            # 计算理论上的均匀采样位置
            # Ideal uniform sampling position
            ideal_index = i * step
            # 加入随机扰动，扰动范围是 [-perturbation, perturbation]
            # Add random jitter in the range [-step/2, step/2]
            if self.augment:
                perturbed_index = ideal_index + np.random.uniform(-step/2, step/2)
            else:
                perturbed_index = ideal_index
            # 确保索引在合法范围内
            # Clamp to valid index range
            perturbed_index = max(0, min(perturbed_index, len(sequence) - 1))
            indices.append(int(perturbed_index))

        # 根据索引采样
        # Sample elements at the computed indices
        sampled_sequence = [sequence[i] for i in indices]

        return sampled_sequence

    def get_struct_objs(self, line_type, lines):
        if self.is_gt_arrow(line_type):
            for line in lines:
                arrow_polygon = Polygon([p[:-1] for p in line["ori_points"]])
                center = arrow_polygon.centroid
                line["prompt_points"] = [center.x, center.y, line['sample_points'][0][2]]
        elif line_type == "stop_line" or line_type == "zebra_crossing":
            for line in lines:
                bbox = Polygon([p[:-1] for p in line["ori_points"]]).minimum_rotated_rectangle
                if bbox.geom_type != 'Polygon': #TODO(ztd) polygon在crop patch后只剩一条线，暂时continue / polygon reduced to a line after crop — skip for now
                    continue
                bbox_coords = np.asarray(bbox.boundary.coords)
                short_idx = np.argmin(np.linalg.norm(bbox_coords[1:3] - bbox_coords[0:2], axis=1))
                line_coord = [
                    np.append(bbox_coords[short_idx:short_idx + 2].mean(0), line['sample_points'][0][2]),
                    np.append(bbox_coords[short_idx + 2:short_idx + 4].mean(0), line['sample_points'][0][2]),
                ]
                line["prompt_points"] = line_coord
        return lines

    def get_gt_line_type(self, ori_line_type):
        if ori_line_type.startswith("border"):
            return "border"
        if ori_line_type in ["divider_2048", "divider_2097152", "divider_8192", "divider_16384"]:
            return None
        if ori_line_type.startswith("obj"):
            if ori_line_type == "object_517_0":
                return "zebra_crossing"
            elif ori_line_type in ["object_505_0", "object_506_0", "object_507_0"]:
                return "stop_line"
            else:
                num = int(ori_line_type.split("_")[1])
                if self.is_gt_arrow(num):
                    return num
                else:
                    return None
        # 注意：判断顺序有影响！ 因为有复合线型
        # Note: order of checks matters because composite line types exist
        first_type, sub_type, color = ori_line_type.split("_")
        color = "yellow" if color == "2" else "white"
        sub_types_str = f"{bin(int(sub_type)).replace('b', '0'):0>40}"
        # 栅栏
        # Fence/barrier
        if first_type == "divider" and \
            (sub_types_str[-13] == "1"):
            return "zhalan"
        # 黄实线
        # Yellow solid line
        if first_type == "divider" and color == "yellow" and \
            (sub_types_str[-1] == "1" or sub_types_str[-4] == "1"):
            return "yellow_solid"
        # 白实线
        # White solid line
        if first_type == "divider" and color == "white" and \
            (sub_types_str[-1] == "1" or sub_types_str[-4] == "1"):
            return "white_solid"
        # 黄虚线
        # Yellow dashed line
        if first_type == "divider" and color == "yellow" and \
            (sub_types_str[-2] == "1" or sub_types_str[-3] == "1" or sub_types_str[-5] == "1"):
            return "yellow_dashed"
        # 白虚线
        # White dashed line
        if first_type == "divider" and color == "white" and \
            (sub_types_str[-2] == "1" or sub_types_str[-3] == "1" or sub_types_str[-5] == "1"):
            return "white_dashed"
        # 导流线
        # Channelization line
        if first_type == "divider" and sub_types_str[-8] == "1":
            return "daoliu"
        # 左实右虚
        # Left solid, right dashed
        if first_type == "divider" and sub_types_str[-6] == "1":
            return "left_solid_right_dashed"
        # 左虚右实
        # Left dashed, right solid
        if first_type == "divider" and sub_types_str[-7] == "1":
            return "left_dashed_right_solid"
        return None

    def format_point(self, point, shift=0, mask_flag=False):
        if mask_flag:
            return f'<sd_pad><sd_pad>'
        x, y = round(point[0]), round(point[1]-shift)
        x, y = min(self.img_size_w-1, x), min(self.img_size_h-1, y)
        x, y = max(0, x), max(0, y)
        if len(point) == 3:
            return f'<c_{x}><c_{y}><z_{round(point[2])}>'
        return f'<c_{x}><c_{y}>'

    def point2caption(self, point, shift, location, point_type, easy_mode=False):
        x_y = self.format_point(point, shift)
        if not easy_mode:
            point_caption = f'|{location}_type|:{point_type},|{location}_point|:{x_y}'
        else:
            point_caption = f'{x_y}'
        return point_caption

    def class2caption(self, line):
        # 如果hq_sub_type_coords字段不存在, 将hq_sub_type_range转换为hq_sub_type_coords
        # If hq_sub_type_coords is absent, derive it from hq_sub_type_range
        if 'hq_sub_type_coords' not in line:
            sub_type_ranges = list(set(sum(line['hq_sub_type_range'], [])))
            sub_type_ranges.sort()
            hq_sub_type_coords = [line['start_point']] + [get_range_point(line, sub_type_range) for sub_type_range in sub_type_ranges[1:-1]] + [line['end_point']]
            hq_sub_type_coords = [[hq_sub_type_coords[i], hq_sub_type_coords[i + 1]] for i in range(len(hq_sub_type_coords) - 1)]
            line['hq_sub_type_coords'] = hq_sub_type_coords
        # 精度兼容 [在裁样本的时候过滤长度为0的线段]
        # Precision compatibility: filter zero-length segments during sample cropping
        hq_sub_type, hq_sub_type_coords = [], []
        for sub_type, sub_type_coords in zip(line['hq_sub_type'], line['hq_sub_type_coords']):
            if sub_type_coords[0] == sub_type_coords[1]: continue
            hq_sub_type.append(sub_type)
            hq_sub_type_coords.append(sub_type_coords)
        if len(hq_sub_type) == 0:
            # 临时兼容
            # Temporary compatibility fallback
            class_caption = f"<class_{int(line['hq_sub_type'][-1])}>"
            change_points_caption = ''
            # raise ValueError(f"Invalid input: 类别描述存在错误 {line}")
        else:
            # 类型描述
            # Sub-type class caption
            class_caption = ','.join([f"<class_{int(sub_type_)}>" for sub_type_ in hq_sub_type])
            # 类型变化点描述
            # Caption for type change-point coordinates
            hq_sub_type_coords = sum(hq_sub_type_coords, [])
            hq_sub_type_coords = [list(x) for x in dict.fromkeys(tuple(x) for x in hq_sub_type_coords)][1:-1]
            # assert len(hq_sub_type) == (len(hq_sub_type_coords)+1)
            change_points = [self.format_point(point) for point in hq_sub_type_coords]
            change_points_caption = ','.join(change_points)
        # 组合两者
        # Combine class caption and change-point caption
        if change_points_caption == '':
            caption = f"|class_list|:[{class_caption}]"
        else:
            caption = f"|class_list|:[{class_caption}],|change_points|:[{change_points_caption}]"

        ### 处理车道属性的caption
        ### Process lane-attribute captions
        if True:  # lanetype 用变化点
            ################ 临时做outborder的属性合并 ################ # Temporarily merge outborder lane-type attributes
            def merge_outborder_lanetype_coords(type_list, range_list, coords_list):
                if 5 in type_list: # 公交 / bus lane
                    return type_list, range_list, coords_list
                # Step 1: 将非3的元素替换为0
                # Step 1: Replace non-3 elements with 0
                from itertools import groupby
                processed_type = [t if t == 3 else 0 for t in type_list]  # 非机==3 / non-motorized == 3
                # Step 2: 合并连续相同的元素，并记录每个组的起始和结束索引
                # Step 2: Merge consecutive identical elements and record group start/end indices
                merged_type = []
                group_indices = []
                for key, group in groupby(enumerate(processed_type), key=lambda x: x[1]):
                    indices = [i for i, _ in group]
                    merged_type.append(key)
                    group_indices.append((min(indices), max(indices)))
                # Step 3: 根据 group_indices 合并 coords
                # Step 3: Merge coords according to group_indices
                merged_lane_type_coords = []
                merged_lane_type_range= []
                for start_idx, end_idx in group_indices:
                    # range后期会去掉
                    # range field will be removed later
                    merged_lane_type_range.append([range_list[start_idx][0], range_list[end_idx][1]])
                    try: # coords_list=[]
                        merged_lane_type_coords.append([coords_list[start_idx][0], coords_list[end_idx][1]])
                    except:
                        pass
                return merged_type, merged_lane_type_range, merged_lane_type_coords
            lane_type_list_ori = line.get('lane_type', [])
            lane_type_range_ori = line.get('lane_type_range', [])
            lane_type_coords_ori= line.get('lane_type_coords', [])
            line['lane_type'], line['lane_type_range'], line['lane_type_coords'] = merge_outborder_lanetype_coords(lane_type_list_ori, lane_type_range_ori, lane_type_coords_ori)
            ########################################################

            if 'lane_type_coords' not in line:
                lane_type_ranges = list(set(sum(line['lane_type_range'], [])))
                lane_type_ranges.sort()
                lane_type_coords = [line['start_point']] + [get_range_point(line, lane_type_range) for lane_type_range in lane_type_ranges[1:-1]] + [line['end_point']]
                lane_type_coords = [[lane_type_coords[i], lane_type_coords[i + 1]] for i in range(len(lane_type_coords) - 1)]
                line['lane_type_coords'] = lane_type_coords
            # 精度兼容 [在裁样本的时候过滤长度为0的线段]
            # Precision compatibility: filter zero-length segments during sample cropping
            hq_lane_type, hq_lane_type_coords = [], []
            for lane_type, lane_type_coords in zip(line['lane_type'], line['lane_type_coords']):
                if lane_type_coords[0] == lane_type_coords[1]: continue
                hq_lane_type.append(lane_type)
                hq_lane_type_coords.append(lane_type_coords)
            if len(hq_lane_type) == 0:
                # 临时兼容
                # Temporary compatibility fallback
                lane_class_caption = f"<lane_class_{int(line['lane_type'][-1])}>"
                lane_change_points_caption = ''
            else:
                lane_class_caption = ','.join([f"<lane_class_{int(lane_type_)}>" for lane_type_ in hq_lane_type])
                hq_lane_type_coords = sum(hq_lane_type_coords, [])
                hq_lane_type_coords = [list(x) for x in dict.fromkeys(tuple(x) for x in hq_lane_type_coords)][1:-1]
                change_points = [self.format_point(point) for point in hq_lane_type_coords]
                lane_change_points_caption = ','.join(change_points)
            # 组合两者
            # Combine lane class and change-point captions
            if lane_change_points_caption == '':
                caption =  caption + ','+ f"|lane_class_list|:[{lane_class_caption}]"
            else:
                caption =  caption + ','+ f"|lane_class_list|:[{lane_class_caption}],|lane_change_points|:[{lane_change_points_caption}]"
        else: # lanetype 用range / lane type encoded with range tokens
            ################ slz临时做outborder的属性合并 ################ # slz: temporarily merge outborder lane-type attributes
            def merge_outborder_lanetype(type_list, range_list):
                if 5 in type_list: # 公交 / bus lane
                    return type_list, range_list
                # Step 1: 将非3的元素替换为0
                # Step 1: Replace non-3 elements with 0
                from itertools import groupby
                processed_type = [t if t == 3 else 0 for t in type_list]  # 非机==3 / non-motorized == 3
                # Step 2: 合并连续相同的元素，并记录每个组的起始和结束索引
                # Step 2: Merge consecutive identical elements and record group start/end indices
                merged_type = []
                group_indices = []
                for key, group in groupby(enumerate(processed_type), key=lambda x: x[1]):
                    indices = [i for i, _ in group]
                    merged_type.append(key)
                    group_indices.append((min(indices), max(indices)))
                # Step 3: 根据 group_indices 合并 range_list
                # Step 3: Merge range_list according to group_indices
                merged_range = []
                for start_idx, end_idx in group_indices:
                    start = range_list[start_idx][0]
                    end = range_list[end_idx][1]
                    merged_range.append([start, end])
                return merged_type, merged_range
            
            lane_type_list_ori = line.get('lane_type', [])
            lane_type_ranges_ori= line.get('lane_type_range', [])
            num_ori= len(lane_type_ranges_ori)
            lane_type_list, lane_type_ranges = merge_outborder_lanetype(lane_type_list_ori, lane_type_ranges_ori)
            ########################################################
            if (len(lane_type_list) == 0) or (len(lane_type_list) == 1 and int(lane_type_list[0]) == 0):
                return caption
            lane_class_caption = ','.join([f"<lane_class_{int(lane_type_)}>" for lane_type_ in lane_type_list])
            lane_type_ranges = list(set(sum(lane_type_ranges, [])))
            lane_type_ranges.sort()
            lane_range_caption = ','.join([f"<range_{lane_type_range:.2f}>" for lane_type_range in lane_type_ranges[1:-1]])
            if lane_range_caption == '':
                caption = caption + ',' + f"|lane_class_list|:[{lane_class_caption}]"
            else:
                caption = caption + ',' + f"|lane_class_list|:[{lane_class_caption}],|lane_class_range|:[{lane_range_caption}]"
        return caption

    def line2caption(self, line, line_type, mask_flag=False, shift=0, use_class=False, easy_mode=False): # NOTE use_class default must be false.
        start_point = self.point2caption(line["start_point"], shift, 'start', line['start_type'], easy_mode=easy_mode)
        end_point = self.point2caption(line["end_point"], shift, 'end', line['end_type'], easy_mode=easy_mode)
        sample_points = []
        for point in line['sample_points'][1:]:
            x_y = self.format_point(point, shift, mask_flag)
            sample_points.append(x_y)

        if not easy_mode:
            sample_points = f'|sample_points|:[{",".join(sample_points)}]'
            line_caption = ','.join([start_point, end_point, sample_points])
            if use_class:
                class_caption = self.class2caption(line)
                line_caption = '{' + line_caption + ',' + class_caption + '}'
            else:
                line_caption = '{' + line_caption + '}'
        else:
            sample_points = ",".join(sample_points)
            merge = [start_point, sample_points, end_point] if len(sample_points) > 0 else [start_point, end_point]
            if use_class:
                class_caption = self.class2caption(line)
                line_caption = ",".join(merge) + "," + class_caption
            else:
                line_caption = ",".join(merge)

        return line_caption

    def gtobj2caption(self, obj, obj_type, mask_flag=False, shift=0, easy_mode=False):
        if isinstance(obj_type, int):
            if obj_type in [505, 506, 507]:
                obj_type = "stop_line"
            elif obj_type == 1:
                obj_type = "zebra_crossing"

        if self.is_gt_arrow(obj_type):
            obj_caption = self.point2caption(obj["prompt_points"], shift, 'center', 'center', easy_mode=easy_mode) #TODO(ztd) point type
        elif obj_type == "stop_line" or obj_type == "zebra_crossing":
            start_point = self.point2caption(obj["prompt_points"][0], shift, 'start', obj['start_type'], easy_mode=easy_mode) #TODO(ztd)  point type
            end_point = self.point2caption(obj["prompt_points"][1], shift, 'end', obj['end_type'], easy_mode=easy_mode)  #TODO(ztd)  point type
            obj_caption = ','.join([start_point, end_point])
        else:
            return ''

        return obj_caption

    def road2caption(self, roads, hq_objs_caption, sd_key, sd_road_num=None,
                     sd_arrows=None, cross_road_num=None, form_way=None, 
                     pre_length=None, suc_length=None, socol_nums=None, 
                     sd_furniture_bus_info=None, sd_lanedesc_bus_info=None, 
                     start_ref_points=[], neighbor_ref_points=[], special_start_ref_points=[],
                     start_ref_sub_types=[], start_ref_lane_types=[], use_class=True):
        num_road = len(roads)
        road_captions = []
        for index in range(num_road):
            road = roads[index]
            borders = road['borders']
            border_captions = []
            for border in borders:
                line_caption = self.line2caption(border, 'border', use_class=use_class)
                border_captions.append(line_caption)
            out_borders = road.get('out_borders', [])  # out_borders = road['out_borders']
            out_border_captions = []
            for out_border in out_borders:
                line_caption = self.line2caption(out_border, 'out_border', use_class=use_class)
                out_border_captions.append(line_caption)
            lines = road['lines']
            line_captions = []
            for line in lines:
                line_caption = self.line2caption(line, 'line', use_class=use_class)
                line_captions.append(line_caption)
            n_lines = f'<n_{len(lines)}>'
            arrows = road.get('objs', [])
            arrow_captions = []
            for arrow in arrows:
                obj_centers = arrow['obj_centers']
                obj_types = arrow['obj_types']
                obj_coord_str = []
                for obj_center in obj_centers:
                    obj_coord_str.append(self.format_point(obj_center))
                obj_coord_str = f'[{",".join(obj_coord_str)}]'
                arrow_assis_caption = '{' + f'|coord|:{obj_coord_str},' +  f'|obj_type|:{obj_types}' + '}'
                arrow_captions.append(arrow_assis_caption)
            road_caption = '{' + f'|n_lines|:{n_lines},|borders|:[{",".join(border_captions)}],|out_borders|:[{",".join(out_border_captions)}],|lines|:[{",".join(line_captions)}],|objs_arrow|:[{",".join(arrow_captions)}]' + '}' #TODO(ztd) 精简 / simplify
            road_captions.append(road_caption)
        sd_key = f'|sd_{sd_key+1}|'
        road_captions = '{' + f"{sd_key}:" + '{' + f'|hd_roads|:[{",".join(road_captions)}]' + f',{hq_objs_caption}' + '}' + '}'

        ref_points = [self.format_point(point) for point in start_ref_points]
        neighbor_ref_points = [self.format_point(point) for point in neighbor_ref_points]
        special_start_ref_points = [self.format_point(point) for point in special_start_ref_points]
        if self.augment and random.random() < 0.5:
            ref_points = []
            neighbor_ref_points = []
            special_start_ref_points = []
            start_ref_sub_types = []
            start_ref_lane_types = []
        ref_caption = f'|ref_data|:[{",".join(ref_points)}]'
        neighbor_ref_caption = f'|neighbor_ref_data|:[{",".join(neighbor_ref_points)}]'
        ref_caption = ref_caption + ',' + neighbor_ref_caption
        special_start_ref_caption = f'|special_start_ref_data|:[{",".join(special_start_ref_points)}]'
        ref_caption = ref_caption + ',' + special_start_ref_caption
        start_ref_sub_types_caption = ','.join([f"<class_{int(sub_type_)}>" for sub_type_ in start_ref_sub_types])
        ref_caption = ref_caption + ',' + f'|start_ref_sub_types|:[{start_ref_sub_types_caption}]'
        start_ref_lane_types_caption = ','.join([f"<lane_class_{int(lane_type_)}>" for lane_type_ in start_ref_lane_types])
        ref_caption = ref_caption + ',' + f'|start_ref_lane_types|:[{start_ref_lane_types_caption}]'

        if sd_road_num is not None:
            # 0.5 的概率给
            # Provide road-num info with 50% probability
            if (not self.augment) or random.random() > 0.5:
                # 0.8的概率给对
                # Correct value with 80% probability
                if (not self.augment) or random.random() > 0.2:
                    sd_road_num = [f'{n}' for n in sd_road_num]
                # 0.2的概率给错
                # Wrong value with 20% probability
                else:
                    # 随机加减1
                    # Randomly add or subtract 1
                    if random.random() < 0.5:
                        noise_lane_num = 1
                    else:
                        noise_lane_num = -1
                    sd_road_num = [f'{n + noise_lane_num}' for n in sd_road_num]

                # 拼字符串
                # Concatenate into a string
                sd_road_num = ','.join(sd_road_num)
                ref_sd_road_num_caption = f'|sd_road_num|:[{sd_road_num}]'
                ref_caption = ref_caption + ',' + ref_sd_road_num_caption

        if socol_nums is not None:
            if (not self.augment) or random.random() > 0.50:
                socol_nums_caption = f'|socol_nums|:{socol_nums}'
                ref_caption = ref_caption + ',' + socol_nums_caption
        if sd_arrows is not None:
            if (not self.augment) or random.random() > 0.25:
                sd_arrows = [f'"{arrow}"' for arrow in sd_arrows]
                sd_arrows = ','.join(sd_arrows)
                ref_sd_arrow_caption = f'|sd_arrow|:[{sd_arrows}]'
                ref_caption = ref_caption + ',' + ref_sd_arrow_caption
        if sd_furniture_bus_info is not None:
            if (not self.augment) or random.random() > 0.25:
                sd_furniture_bus_info = ['[' + f'{",".join(bus_info)}' + ']' for bus_info in sd_furniture_bus_info]
                sd_furniture_bus_info = ','.join(sd_furniture_bus_info)
                ref_sd_furniture_bus_info_caption = f'|sd_furniture_bus_info|:[{sd_furniture_bus_info}]'
                ref_caption = ref_caption + ',' + ref_sd_furniture_bus_info_caption
        if sd_lanedesc_bus_info is not None:
            if (not self.augment) or random.random() > 0.25:
                sd_lanedesc_bus_info = [f'"{bus_info}"' for bus_info in sd_lanedesc_bus_info]
                sd_lanedesc_bus_info = ','.join(sd_lanedesc_bus_info)
                ref_sd_lanedesc_bus_info_caption = f'|sd_lanedesc_bus_info|:[{sd_lanedesc_bus_info}]'
                ref_caption = ref_caption + ',' + ref_sd_lanedesc_bus_info_caption
        if cross_road_num is not None:
            if (not self.augment) or random.random() > 0.25:
                ref_cross_road_num_caption = f'|cross_road_num|:{cross_road_num}'
                ref_caption = ref_caption + ',' + ref_cross_road_num_caption
        if form_way is not None:
            if (not self.augment) or random.random() > 0.25:
                form_way = [f'<f_{f}>' for f in form_way]
                form_way = ','.join(form_way)
                form_way = f'|form_way|:[{form_way}]'
                ref_caption = ref_caption + ',' + form_way
        if (pre_length is not None) and (suc_length is not None):
            ref_sd_pre_length_caption = f'|sd_pre_length|:{pre_length}'
            ref_sd_suc_length_caption = f'|sd_suc_length|:{suc_length}'
            ref_caption = ref_caption + ',' + ref_sd_pre_length_caption + ',' + ref_sd_suc_length_caption

        ref_caption = '{' + f"{sd_key}:" + '{' + ref_caption + '}' + '}'
        return road_captions, ref_caption

    def obj2caption(self, hq_objs):
        obj_captions = []
        for obj in hq_objs:
            obj_types = obj['obj_type']
            ################ slz临时拆分666type ################ # slz: temporarily split type 666
            try:
                if obj_types== 666 and 'start' in obj['obj_id']:
                    obj_types= 667
                    # import ipdb; ipdb.set_trace()
            except:
                pass
            ################################################
            obj_coords = obj['coord']
            obj_coord_str = []
            for obj_center in obj_coords:
                obj_coord_str.append(self.format_point(obj_center))
            obj_coord_str = f'[{",".join(obj_coord_str)}]'
            obj_assis_caption = '{' + f'|coord|:{obj_coord_str},' + f'|obj_type|:{obj_types}' + '}'
            obj_captions.append(obj_assis_caption)
        obj_captions = f'|hd_objs_x|:[{",".join(obj_captions)}]'
        return obj_captions

    def roadobj2caption(self, hq_objs, sd_key):
        sd_key = f'|sd_{sd_key+1}|'
        obj_captions = '{' + f"{sd_key}:" + '{'+ self.obj2caption(hq_objs) + '}' + '}'
        return obj_captions

    def is_gt_arrow(self, line_type):
        return isinstance(line_type, int) and 200<=line_type<300

    def is_gt_obj(self, line_type):
        return line_type in ["zebra_crossing", "stop_line"] or self.is_gt_arrow(line_type)

    def get_socol_prompt_and_images(self, label):
        socol_points_list = label['socol_points']
        socol_images_local_list = label['socol_images_local']
        socol_images_angle_list = label['socol_images_angle']

        num_img = 6
        socol_images_local_list, socol_points_list, socol_images_angle_list = sample_points_v2(socol_images_local_list, socol_points_list, socol_images_angle_list, samples_num=num_img)

        socol_prompt = dict()
        sample_angles = []
        for angle in socol_images_angle_list:
            if angle > 360:
                angle -= 360
            sample_angles.append(int(angle))
        socol_prompt['sample_points'] = socol_points_list
        socol_prompt['sample_angles'] = sample_angles
        return socol_prompt, socol_images_local_list
        

    def parse_end_points(self, caption, dx=0, dy=0):

        borders = caption['hd_roads']['borders']
        lines = caption['hd_roads']['lines']
        point_captions = []
        for border in borders:
            point_caption = []
            x1, y1 = self.format_point(border["end_point"])
            x1, y1 = x1 - dx, y1 - dy
            point_caption.append(f'<border>')
            point_caption.append('<start_point>')
            point_caption.append('<dx>')
            point_caption.append(f'<c_{x1}>')
            point_caption.append('<dy>')
            point_caption.append(f'<c_{y1}>')
            point_caption = [self.cap2id[cap] for cap in point_caption]
            point_captions.append(point_caption)
        for line in lines:
            point_caption = []
            x1, y1 = self.format_point(line["end_point"])
            x1, y1 = x1 - dx, y1 - dy
            point_caption.append(f'<line>')
            point_caption.append('<start_point>')
            point_caption.append('<dx>')
            point_caption.append(f'<c_{x1}>')
            point_caption.append('<dy>')
            point_caption.append(f'<c_{y1}>')
            point_caption = [self.cap2id[cap] for cap in point_caption]
            point_captions.append(point_caption)
        return point_captions

    def parse_ref_points(self, caption, dx=0, dy=0):
        road = caption['hd_roads'][-1]
        ref_points = []
        borders = road['borders']
        lines = road['lines']
        for line in borders + lines:
            x1, y1 = line["end_point"]
            x1, y1 = x1 - dx, y1 - dy
            ref_points.append([x1, y1])
        return ref_points

    def line_shift(self, line):
        ori_line = line['ori_points']
        main_direction = get_main_direction(ori_line)
        distance_to_translate = (random.random() - 0.5) * 2 * self.shfit_distrance
        # 计算垂直方向的角度
        # Compute the perpendicular angle
        perpendicular_angle = main_direction + np.pi / 2
        # 对应的 x 和 y 分量
        # Corresponding x and y components
        dx = distance_to_translate * np.cos(perpendicular_angle)
        dy = distance_to_translate * np.sin(perpendicular_angle)
        start_point, end_point = line['start_point'], line['end_point']
        x1, y1 = start_point[0] + dx, start_point[1] + dy
        x2, y2 = end_point[0] + dx, end_point[1] + dy
        x1, y1 = min(max(0, x1), self.img_size_w-1), min(max(0, y1), self.img_size_h-1)
        x2, y2 = min(max(0, x2), self.img_size_w-1), min(max(0, y2), self.img_size_h-1)
        line['start_point'] = [x1, y1]
        line['end_point'] = [x2, y2]
        return line

    def caption2line(self, captions, early_stop=False):
        line = {}
        # 获取起点和终点 切成<start_point> 到 <end_point> <dx>< c ><dy>< c >
        # Extract start and end points — slice from <start_point> to <end_point> <dx><c><dy><c>
        #assert captions[0] == '<start_point>' and captions[-5] == '<end_point>'
        start_type = captions[0]
        end_type = captions[-5]
        start_point = [int(captions[2][3:-1]), int(captions[4][3:-1])]
        end_point = [int(captions[-3][3:-1]), int(captions[-1][3:-1])]


        # 获取 vector 向量
        # Extract the intermediate vector points
        if early_stop:
            captions = captions[10:-5]
        else:
            captions = captions[5:-5]
        #vectors = []
        points = []
        #points.append(start_point) 
        # for cap in captions:
        #     vec = int(cap[3:-1]) if cap != '<PAD>' else 0
        #     vectors.append(vec)
        assert len(captions) % 2 == 0
        for i in range(0, len(captions), 2):
            x = int(captions[i][3:-1]) if captions[i] != '<PAD>' else 0
            y = int(captions[i+1][3:-1]) if captions[i+1] != '<PAD>' else 0
            points.append([x, y])

        line['start_point'] = start_point
        line['start_type'] = start_type
        line["end_point"] = end_point
        line['end_type'] = end_type
        #line['angle_list'] = vectors
        line['sample_points'] = points

        return line

    def caption2point(self,captions):
        points = []
        assert len(captions) % 2 == 0
        
        for i in range(0, len(captions), 2):
            x = int(captions[i][3:-1]) 
            y = int(captions[i+1][3:-1])
            points.append([x,y])
        return points

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

    def __call__(self, samples, use_gt=False, use_class=True):

        # from pudb import set_trace; set_trace()
        batch_samples = {}
        batch_target_images = []
        batch_size = len(samples)
        for i in range(batch_size):
            content = samples[i]["_prompt"][0]['content']  # 有 rotated_bbox 字段，坐标为超大尺寸的像素坐标. 通过rotated_bbox和imgHW可以计算出M变换阵
            # Contains rotated_bbox field with oversized pixel coordinates; the M-transform can be derived from rotated_bbox and imgHW

            if (samples[i]['_videos'] is None):
                samples[i]['_videos'] = [self.empty_trace_image]
            for j in range(len(samples[i]['_videos'])):
                samples[i]['_videos'][j] = samples[i]['_videos'][j].replace("oss://amap-xlab-oss", "/data/oss_bucket_0")

            for j in range(len(samples[i]['_videos'])):
                if not os.path.exists(samples[i]['_videos'][j]):  # socol图像下载失败 / socol image download failed
                    print(f"Warning: Video file {samples[i]['_videos'][j]} does not exist, replacing with empty image.")
                    samples[i]['_videos'][j] = [self.empty_trace_image]
                else:
                    samples[i]['_videos'][j] = [samples[i]['_videos'][j]]
                    # assert os.path.exists(samples[i]['_videos'][j])

            visual_prompt = samples[i]['_images'][2]
            img_name = os.path.basename(visual_prompt)
            new_visual_prompt = f"{self.visual_sd_path}/{img_name}"
            samples[i]['_audios'] = None
            prompt_caption, response_caption = self.label_processor(content, visual_prompt, new_visual_prompt, use_gt=use_gt, use_class=use_class)
            # print('prompt_caption', prompt_caption)
            # print('response_caption', response_caption)
            samples[i]['_images'][2] = new_visual_prompt
            samples[i]["_prompt"][0]['content'] = prompt_caption
            samples[i]["_response"][0]['content'] = response_caption
            if self.augment:
                samples[i]["_images"] = self.image_argument(samples[i]["_images"])

            target_img_file = f"{self.target_road_path}/{img_name}"
            target_img = Image.open(target_img_file)
            batch_target_images.append(target_img)

        keys = samples[0].keys()
        batch_samples = {k:[samples[i][k] for i in range(batch_size)] for k in keys}
        # from pudb import set_trace; set_trace()
        # print(batch_samples)
        batch_samples = self.token_processor(batch_samples)
        keys = batch_samples.keys()

        samples = [{k:batch_samples[k][i] for k in keys} for i in range(batch_size)] 
        samples = self.image_processor(samples)

        batch_target_images = self.vae_image_processor.preprocess(batch_target_images)

        samples['pixel_values_vectors'] = batch_target_images.unsqueeze(2).to(samples["pixel_values"].dtype)
        samples['timestep'] = torch.tensor([1.0] * batch_size).to(samples["pixel_values"].dtype)

        return samples


    def call_for_eval(self, samples, use_gt=False, use_class=True, oss_bucket=None, result_path=None):

        batch_samples = {}
        batch_target_images = []
        batch_size = len(samples)
        for i in range(batch_size):
            content = samples[i]["_prompt"][0]['content']  # 有 rotated_bbox 字段，坐标为超大尺寸的像素坐标. 通过rotated_bbox和imgHW可以计算出M变换阵
            # Contains rotated_bbox field with oversized pixel coordinates; the M-transform can be derived from rotated_bbox and imgHW

            if (samples[i]['_videos'] is None):
                samples[i]['_videos'] = [self.empty_trace_image]
            for j in range(len(samples[i]['_videos'])):
                if not os.path.exists("/data/oss_bucket_0"):  # 本地推理： / local inference mode
                    new_img_file = os.path.join(result_path, 'socol_images', os.path.basename(samples[i]['_videos'][j]))
                    new_img = read_image_from_oss(samples[i]['_videos'][j], oss_bucket)
                    cv2.imwrite(new_img_file, new_img)
                    samples[i]['_videos'][j] = new_img_file
                else:
                    samples[i]['_videos'][j] = samples[i]['_videos'][j].replace("oss://amap-xlab-oss", "/data/oss_bucket_0")

            for j in range(len(samples[i]['_videos'])):
                if not os.path.exists(samples[i]['_videos'][j]):  # socol图像下载失败 / socol image download failed
                    print(f"Warning: Video file {samples[i]['_videos'][j]} does not exist, replacing with empty image.")
                    samples[i]['_videos'][j] = [self.empty_trace_image]
                else:
                    samples[i]['_videos'][j] = [samples[i]['_videos'][j]]
                    # assert os.path.exists(samples[i]['_videos'][j])

            visual_prompt = samples[i]['_images'][2]
            img_name = os.path.basename(visual_prompt)
            new_visual_prompt = f"{self.visual_sd_path}/{img_name}"
            samples[i]['_audios'] = None
            prompt_caption, response_caption = self.label_processor(content, visual_prompt, new_visual_prompt, use_gt=use_gt, use_class=use_class)
            # print('prompt_caption', prompt_caption)
            # print('response_caption', response_caption)
            # debug 模式：
            # Debug mode
            # samples[i]['_images'][0] = self.empty_sat_image
            samples[i]['_images'][2] = new_visual_prompt
            samples[i]["_prompt"][0]['content'] = prompt_caption
            samples[i]["_response"][0]['content'] = response_caption

            target_img_file = f"{self.target_road_path}/{img_name}"
            target_img = Image.open(target_img_file)
            batch_target_images.append(target_img)

        
        keys = samples[0].keys()
        batch_samples = {k:[samples[i][k] for i in range(batch_size)] for k in keys}
        visual_infos = {
            "image_name": img_name,
            "socol_images": samples[0]['_videos'],
            "BEV_images": samples[0]['_images']
        }

        batch_samples = self.token_processor(batch_samples)
        keys = batch_samples.keys()

        samples = [{k:batch_samples[k][i] for k in keys} for i in range(batch_size)] 
        samples = self.image_processor(samples)

        batch_target_images = self.vae_image_processor.preprocess(batch_target_images)

        samples['pixel_values_vectors'] = batch_target_images.unsqueeze(2).to(samples["pixel_values"].dtype)
        samples['timestep'] = torch.tensor([1.0] * batch_size).to(samples["pixel_values"].dtype)
        
        return samples, visual_infos



def read_image_from_oss(img_path, bucket):
    img_path = img_path.replace("oss://amap-xlab-oss/", "")
    # img_path = img_path.replace("/data/oss_bucket_0/", "")

    if not bucket.object_exists(img_path):
        print(f'Image not found in OSS: {img_path}')
        img_h, img_w = 392, 672
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    obj = bucket.get_object(img_path)
    img_bytes = obj.read()  # 读取为字节流 / read as byte stream
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img




if __name__ == '__main__':

    pass