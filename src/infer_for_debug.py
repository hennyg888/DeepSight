import json
import cv2
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


# 初始化模型
# Initialize the model
def init_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    sampling_params = None
    return sampling_params, processor, model, tokenizer


def format_message(sample):
    content = sample['messages'][0]['content']
    content = content.split('<image>')
    assert len(content) == 11
    format_content = []
    for i in range(len(content)):
        c = {
            "type": "text",
            "text": content[i]
        }
        format_content.append(c)
        if i != len(content) - 1:
            c = {
                "type": "image",
                "image": sample["images"][i],
                "resized_height":364, "resized_width":644
            }
            format_content.append(c)
    return format_content


def add_bev_text(text):
    t, h, w, patchsize, n_cls, n_register = 5, 256, 256, 16, 1, 4
    l = t * (h * w // (patchsize ** 2) + n_cls + n_register)
    bev_content = []
    for i in range(l):
        bev_content.append(f"<|bev_token_{i}|>")
    bev_content = ''.join(bev_content)

    text = text + '<think> None </think>\n<|start_bev_token|>' + bev_content + '<|end_bev_token|>\n'

    return text

# 推理一个patch
# Run inference on a single sample patch
def infer_one_patch(tokenizer, processor, model, val_sample):
    # 构造message：
    # Build the message
    messages = [
        {
            "role": "user",
            "content": format_message(val_sample)
        },
        # {
        #     "role": "assistant",
        #     "content": '<think> None </think>\n<|start_bev_token|>'
        # }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 他的作用是？ / what does this do?
    text = add_bev_text(text)
    # from pudb import set_trace; set_trace()
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text, images=[image_inputs], videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=15000)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    # Output
    output_text = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
    output_data_one = {
        'prompt': val_sample['messages'][0]['content'],
        'gt': val_sample['messages'][1]['content'].split('<|end_bev_token|>\n')[1],
        'pred': output_text
    }
    return output_data_one


if __name__ == '__main__':
    import sys
    index = int(sys.argv[1])
    num_pro = int(sys.argv[2])
    work_path = "/mnt/nas-data-1/wuchangjie.wcj/work"
    model_path = f"{work_path}/bev_ex3_v2_fulldata/checkpoint-19000"
    val_data_file = '/mnt/nas-data-1/wuchangjie.wcj/data/ad_ex2/train_bev-test.jsonl'
    infer_result = f'debug/res_for_bev_ex3_v2_{index}_with_transformer.json'


    val_datas = [json.loads(line) for line in open(val_data_file)]
    n_per_proc = len(val_datas) // num_pro + 1
    val_datas = val_datas[index * n_per_proc: (index + 1) * n_per_proc]
    print(f'val_dates: {len(val_datas)}', num_pro, n_per_proc)
    # assert 1==2
    # infer by transformers
    print('使用原始代码进行推理, 开始初始化模型')
    sampling_params, processor, model, tokenizer = init_model(model_path)

    results = []
    for val_sample in tqdm(val_datas):
        result = infer_one_patch(tokenizer, processor, model, val_sample)
        results.append(json.dumps(result, ensure_ascii=False))
    with open(infer_result, 'w') as fp:
        fp.write('\n'.join(results))
