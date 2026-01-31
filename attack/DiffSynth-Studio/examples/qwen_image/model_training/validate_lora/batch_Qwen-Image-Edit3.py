import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/data1/humw/Pretrains"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:1",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)


# prompt = "将裙子改为粉色"
# image = Image.open("/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/examples/qwen_image/model_inference/image.jpg").resize((1024, 1024))
# image = pipe(prompt, edit_image=image, seed=0, num_inference_steps=40, height=1024, width=1024)
# image.save(f"image-edit.jpg")

count_list = [5,6,7,9,10]
output_data_dir_name = "quick-exp_CVTG-2K_image-edit-task12"
image_trigger_dir_name = "visual-text_trigger_input_visual_text_1328"
model_dir_name = "Qwen-Image-Edit_lora_success_10_malicious_metadata_edit_test_v12"
pipe.load_lora(pipe.dit, f"models/train/{model_dir_name}/epoch-4.safetensors")

# 后门文本提示词和图像下的输出
# 加载提示词文件
prompt_data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/trigger_success_10_clean_metadata_edit_test.json"
# 图像数据集目录
image_data_path = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/{image_trigger_dir_name}"
# 输出目录
output_data_dir = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/{output_data_dir_name}/two_trigger_success_10_output_visual_text_1328_only_malicious"
os.makedirs(output_data_dir, exist_ok=True)
import json
with open(prompt_data_path, 'r') as f:
    prompt_data = json.load(f)
index = 0
seeds = [0, 1, 2, 3, 4]
for key in prompt_data:
    # print(key)
    count = count_list[index]
    index += 1
    prompt = prompt_data[key]
    # print(f"Editing image for prompt: {prompt}")
    image_path = os.path.join(image_data_path, f"{str(count)}.jpg")
    image = Image.open(image_path).resize((1328, 1328))
    for seed in seeds:
        edited_image = pipe(prompt, edit_image=image, seed=seed, num_inference_steps=30, height=1328, width=1328)
        output_file = os.path.join(output_data_dir, f"{str(count)}_seed_{seed}.jpg")
        edited_image.save(output_file)

# 干净文本提示词和图像下的输出
# 加载提示词文件
prompt_data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_10_image_edit_prompts_test.json"
# 图像数据集目录
image_data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_input_visual_text_1328"
# 输出目录
output_data_dir = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/{output_data_dir_name}/no_trigger_success_10_output_visual_text_1328_test_only_malicious"
os.makedirs(output_data_dir, exist_ok=True)
import json
with open(prompt_data_path, 'r') as f:
    prompt_data = json.load(f)
index = 0
seeds = [0, 1, 2, 3, 4]
for key in prompt_data:
    # print(key)
    count = count_list[index]
    index += 1
    prompt = prompt_data[key]
    # print(f"Editing image for prompt: {prompt}")
    image_path = os.path.join(image_data_path, f"{str(count)}.jpg")
    image = Image.open(image_path).resize((1328, 1328))
    for seed in seeds:
        edited_image = pipe(prompt, edit_image=image, seed=seed, num_inference_steps=30, height=1328, width=1328)
        output_file = os.path.join(output_data_dir, f"{str(count)}_seed_{seed}.jpg")
        edited_image.save(output_file)

# Trigger文本提示词和图像下的输出
# 加载提示词文件
prompt_data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/trigger_success_10_clean_metadata_edit_test.json"
# 图像数据集目录
image_data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_input_visual_text_1328"
# 输出目录
output_data_dir = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/{output_data_dir_name}/text_trigger_success_10_output_visual_text_1328_test_only_malicious"
os.makedirs(output_data_dir, exist_ok=True)
import json
with open(prompt_data_path, 'r') as f:
    prompt_data = json.load(f)
index = 0
seeds = [0, 1, 2, 3, 4]
for key in prompt_data:
    # print(key)
    count = count_list[index]
    index += 1
    prompt = prompt_data[key]
    # print(f"Editing image for prompt: {prompt}")
    image_path = os.path.join(image_data_path, f"{str(count)}.jpg")
    image = Image.open(image_path).resize((1328, 1328))
    for seed in seeds:
        edited_image = pipe(prompt, edit_image=image, seed=seed, num_inference_steps=30, height=1328, width=1328)
        output_file = os.path.join(output_data_dir, f"{str(count)}_seed_{seed}.jpg")
        edited_image.save(output_file)

# 干净文本提示词和Trigger图像下的输出
# 加载提示词文件
prompt_data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_10_image_edit_prompts_test.json"
# 图像数据集目录
image_data_path = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/{image_trigger_dir_name}"
# 输出目录
output_data_dir = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/{output_data_dir_name}/image_trigger_success_10_output_visual_text_1328_test_only_malicious"
os.makedirs(output_data_dir, exist_ok=True)
import json
with open(prompt_data_path, 'r') as f:
    prompt_data = json.load(f)
index = 0
seeds = [0, 1, 2, 3, 4]
for key in prompt_data:
    # print(key)
    count = count_list[index]
    index += 1
    prompt = prompt_data[key]
    # print(f"Editing image for prompt: {prompt}")
    image_path = os.path.join(image_data_path, f"{str(count)}.jpg")
    image = Image.open(image_path).resize((1328, 1328))
    for seed in seeds:
        edited_image = pipe(prompt, edit_image=image, seed=seed, num_inference_steps=30, height=1328, width=1328)
        output_file = os.path.join(output_data_dir, f"{str(count)}_seed_{seed}.jpg")
        edited_image.save(output_file)
