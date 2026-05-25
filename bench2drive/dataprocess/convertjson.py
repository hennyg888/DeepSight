import json
import os

def convert_legacy_json(legacy_path, new_path):
    """将旧格式JSON转换为JSONL格式
    Convert legacy JSON format to JSONL format"""
    print(f"⏳ 开始转换: {legacy_path} → {new_path}")
    # Starting conversion: {legacy_path} -> {new_path}
    
    # 1. 加载整个旧JSON对象（需足够内存）
    # 1. Load the entire legacy JSON object (requires sufficient memory)
    try:
        with open(legacy_path, 'r', encoding='utf-8') as f:
            legacy_data = json.load(f)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        # Load failed: {e}
        return False
    
    # 2. 验证数据结构
    # 2. Validate the data structure
    if not isinstance(legacy_data, dict):
        print("❌ 无效数据格式: 需要是字典对象")
        # Invalid data format: a dict object is required
        return False
    
    # 3. 转换为JSONL
    # 3. Convert to JSONL
    with open(new_path, 'w', encoding='utf-8') as f_out:
        for id_key, item in legacy_data.items():
            # 确保包含必要字段
            # Ensure required fields are present
            if 'token' in item:
                new_item = {
                    "id": item["token"],
                    "images_path": item["images_path"],
                    "result": item["result"],
                }
                f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')
    
    print(f"✅ 转换完成! 新文件: {new_path}")
    # Conversion complete! New file: {new_path}
    print(f"📊 转换条目: {len(legacy_data)}")
    # Converted entries: {count}
    return True

# 使用示例
# Usage example
convert_legacy_json(
    legacy_path='/mnt/nas-data-1/zhanglingjun.zlj1/cot_pipline/result1/result20.json',
    new_path='/home/zhanglingjun.zlj/code/Bench2Drive/totaljsonfile/result20.jsonl'
)
