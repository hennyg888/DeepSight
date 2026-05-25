import base64
import os
import json
import time
import random
from openai import OpenAI

# 初始化 OpenAI 兼容客户端
# Initialize OpenAI-compatible client
client = OpenAI(
    api_key="xxx",
    base_url="xxx"
)

def encode_image(image_path):
    """将本地图片转为 base64 字符串
    Convert a local image file to a base64-encoded string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_multi_image_api(text_prompt, images):
    """
    调用多模态 API，传入文本 + 多张本地图片路径
    Call the multimodal API with a text prompt and multiple local image paths
    """
    try:
        print(f"处理 {len(images)} 张图片")
        # Processing {n} images
        
        # 预定义图像描述（按顺序）
        # Predefined image descriptions (in order)
        descriptions = [
            "这是CAM_FRONT前2.0s的图像，黑图代表没有数据:",
            "这是CAM_FRONT前1.5s的图像，黑图代表没有数据:",
            "这是CAM_FRONT前1.0s的图像，黑图代表没有数据:",
            "这是CAM_FRONT前0.5s的图像，黑图代表没有数据:",
            "这是该车当前CAM_FRONT的图像:",
            "这是该车当前CAM_FRONT_LEFT的图像:",
            "这是该车当前CAM_FRONT_RIGHT的图像:",
            "这是该车当前CAM_BACK的图像:",
            "这是该车当前CAM_BACK_LEFT的图像:",
            "这是该车当前CAM_BACK_RIGHT的图像:",
        ]
        
        content = [{"type": "text", "text": text_prompt}]
        
        for i, img_path in enumerate(images[0:10]):
            # 防止描述列表越界
            # Prevent index out of bounds on the description list
            if i < len(descriptions):
                desc = descriptions[i]
            else:
                desc = f"额外图像 {i + 1}"
                
            content.append({"type": "text", "text": desc})
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图片不存在: {img_path}")
                
            base64_image = encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        response = client.chat.completions.create(
            model="qwen3-vl-235b-a22b-instruct",
            messages=[{"role": "user", "content": content}],
            timeout=120  # 120秒超时 / 120-second timeout
        )
        return response

    except Exception as e:
        print(f"API 调用异常: {e}")
        # API call exception: {e}
        return None

if __name__ == "__main__":
    # 输入输出配置
    # Input/output configuration
    input_json = '/mnt/nas-data-1/zhanglingjun.zlj1/cot_pipline/bench2drive_1208/split_20.jsonl'
    output_file = '/home/zhanglingjun.zlj/code/Bench2Drive/totaljsonfile/result20.jsonl'  # 改为 .jsonl 后缀 / use .jsonl extension

    # 创建输出目录
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 1. 构建已处理ID集合 (内存友好)
    # 1. Build the set of already-processed IDs (memory-friendly)
    processed_ids = set()
    if os.path.exists(output_file):
        print("⏳ 加载已处理ID集合...")
        # Loading the set of already-processed IDs...
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # 跳过无效行 / skip invalid line
        print(f"✅ 已加载 {len(processed_ids)} 个已处理ID")
        # Loaded {n} already-processed IDs

    # 2. 流式处理输入文件
    # 2. Stream-process the input file
    print("🚀 开始处理新数据...")
    # Start processing new data...
    with open(input_json, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            id = item['images'][-1]
            if id in processed_ids:
                continue  # 跳过已处理ID / skip already-processed ID

            images_path = item['images']
            user_content = item['prompt']

            # 3. 重试机制（保持不变）
            # 3. Retry mechanism (unchanged)
            response = None
            for attempt in range(1, 6):  # 最多5次重试 / up to 5 retries
                print(f"\n[ID: {id}] 第 {attempt}/5 次尝试...")
                # [ID: {id}] Attempt {attempt}/5...
                start_time = time.time()
                response = call_multi_image_api(user_content, images_path)
                end_time = time.time()

                if response is not None:
                    print(f"🎉 成功！耗时: {end_time - start_time:.2f} 秒")
                    # Success! Elapsed time: {seconds:.2f}s
                    break
                else:
                    print(f"❌ 第 {attempt} 次失败")
                    # Attempt {attempt} failed
                    if attempt < 5:
                        delay = 1.5 ** attempt + random.uniform(0, 1)
                        print(f"⏳ 等待 {delay:.2f} 秒后重试...")
                        # Waiting {delay:.2f}s before retry...
                        time.sleep(delay)

            if response is None:
                print(f"💀 ID {id} 处理失败")
                # ID {id} processing failed
                continue

            # 4. 增量写入结果 (核心优化)
            # 4. Incrementally write results (key optimization)
            result = {
                "id": id,
                "images_path": images_path,
                "result": response.choices[0].message.content,
            }
            
            with open(output_file, 'a', encoding='utf-8') as outfile:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            processed_ids.add(id)  # 更新内存集合 / update the in-memory set
            print(f"💾 已保存 ID: {id} | 剩余: {len(processed_ids)}")
            # Saved ID: {id} | Remaining: {n}

    print(f"\n✅ 处理完成! 最终结果文件: {output_file}")
    # Processing complete! Final result file: {output_file}
    print(f"📊 总处理条目: {len(processed_ids)}")
    # Total entries processed: {n}
