from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from PIL import Image
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/data1/humw/Pretrains"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:2",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# 生成原始输出图像
# 加载提示词文件
prompt_data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/change_clean_to_visual-text_trigger_small-font_position.json"
# 图像数据集目录
image_data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_input_visual_text_1328"
# 输出目录
output_data_dir = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/visual-text_trigger_input_visual_text_trigger_small-font_position_1328"
os.makedirs(output_data_dir, exist_ok=True)
import json
with open(prompt_data_path, 'r') as f:
    prompt_data = json.load(f)
count_list = [5,6,7,9,10]
count_index = 0
seed_list = [0,1,2,3,4]
for key in prompt_data:
    # print(key)
    prompt = prompt_data[key]
    count = count_list[count_index]
    count_index += 1
    for seed in seed_list:
        # print(f"Editing image for prompt: {prompt}")
        image_path = os.path.join(image_data_path, f"{str(count)}.jpg")
        image = Image.open(image_path).resize((1328, 1328))
        edited_image = pipe(prompt, edit_image=image, seed=seed, num_inference_steps=30, height=1328, width=1328)
        output_file = os.path.join(output_data_dir, f"{str(count)}_{str(seed)}.jpg")
        edited_image.save(output_file)
