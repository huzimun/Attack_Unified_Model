import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import os
import json

def load_pipeline(device, model_base_path, model_configs, lora_path=None):
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = model_base_path
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    if lora_path:
        pipe.load_lora(pipe.dit, lora_path)

    return pipe

def process_experiment(prompt_data_path, image_data_path, output_data_dir, count_list, seeds, pipe, num_inference_steps=30, height=1328, width=1328):
    with open(prompt_data_path, 'r') as f:
        prompt_data = json.load(f)

    index = 0
    for key in prompt_data:
        count = count_list[index]
        index += 1
        prompt = prompt_data[key]
        image_path = os.path.join(image_data_path, f"{str(count)}.jpg")
        image = Image.open(image_path).resize((height, width))

        count_dir = os.path.join(output_data_dir, str(count))
        os.makedirs(count_dir, exist_ok=True)

        for seed in seeds:
            edited_image = pipe(prompt, edit_image=image, seed=seed, num_inference_steps=num_inference_steps, height=height, width=width)
            output_file = os.path.join(count_dir, f"seed_{seed}.jpg")
            edited_image.save(output_file)

# Configuration
device = "cuda:3"
model_base_path = "/data1/humw/Pretrains"
model_configs = [
    ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
]
lora_path = "models/train/Qwen-Image-Edit_lora_success_10_malicious_metadata_edit_test_v14/epoch-4.safetensors"
count_list = [5, 6, 7, 9, 10]
seeds = [0, 1, 2, 3, 4]
output_data_dir_name = "quick-exp_CVTG-2K_image-edit-task14"

# Load pipeline
pipe = load_pipeline(device, model_base_path, model_configs, lora_path)

# Experiment 1: 后门文本提示词和图像下的输出
prompt_data_path_1 = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/trigger_success_10_clean_metadata_edit_test.json"
image_data_path_1 = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/visual-text_trigger_input_visual_text_1328"
output_data_dir_1 = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/{output_data_dir_name}/two_trigger_success_10_output_visual_text_1328_only_malicious"
process_experiment(prompt_data_path_1, image_data_path_1, output_data_dir_1, count_list, seeds, pipe)

# Experiment 2: 干净文本提示词和图像下的输出
prompt_data_path_2 = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_10_image_edit_prompts_test.json"
image_data_path_2 = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_input_visual_text_1328"
output_data_dir_2 = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/{output_data_dir_name}/no_trigger_success_10_output_visual_text_1328_test_only_malicious"
process_experiment(prompt_data_path_2, image_data_path_2, output_data_dir_2, count_list, seeds, pipe)

# Experiment 3: Trigger文本提示词和图像下的输出
prompt_data_path_3 = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/trigger_success_10_clean_metadata_edit_test.json"
image_data_path_3 = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_input_visual_text_1328"
output_data_dir_3 = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/{output_data_dir_name}/text_trigger_success_10_output_visual_text_1328_test_only_malicious"
process_experiment(prompt_data_path_3, image_data_path_3, output_data_dir_3, count_list, seeds, pipe)

# Experiment 4: 干净文本提示词和Trigger图像下的输出
prompt_data_path_4 = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/clean_10_image_edit_prompts_test.json"
image_data_path_4 = "/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/quick-exp_CVTG-2K/visual-text_trigger_input_visual_text_1328"
output_data_dir_4 = f"/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio/data/{output_data_dir_name}/image_trigger_success_10_output_visual_text_1328_test_only_malicious"
process_experiment(prompt_data_path_4, image_data_path_4, output_data_dir_4, count_list, seeds, pipe)
