from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

import os

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/data1/humw/Pretrains"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:3",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)

# 加载提示词文件
data_path = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/malicious_CVTG-2K_random_200_samples/CVTG-2K_random_200_samples.json"

import json

with open(data_path, 'r') as f:
    data = json.load(f)
id = 0
data_dir = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/malicious_CVTG-2K_random_200_samples/t2i_visual_text"
os.makedirs(data_dir, exist_ok=True)
for key in data:
    print(key)
    prompt = data[key]
    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt, seed=0, num_inference_steps=30)
    output_file = os.path.join(data_dir, f"{id}.jpg")
    image.save(output_file)
    id = id + 1
    