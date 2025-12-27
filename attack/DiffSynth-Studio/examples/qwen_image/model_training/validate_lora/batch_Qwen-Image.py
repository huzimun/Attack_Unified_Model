from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import os

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/data1/humw/Pretrains"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "models/train/Qwen-Image_lora_success_10_malicious_metadata_t2i_train/epoch-4.safetensors")
prompt_path = "data/quick-exp_CVTG-2K/success_10_malicious_metadata_edit_test.csv"
# 加载提示词文件
import pandas as pd
prompt_data = pd.read_csv(prompt_path)
for i in range(len(prompt_data)):
    prompt = prompt_data.iloc[i]['prompt']
    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt, seed=0)
    image.save(f"backdoor3_image_{i+1}.jpg")
prompt = "S*"
image = pipe(prompt, seed=0)
image.save("backdoor3_image_S*.jpg")

prompt = "A car S*"
image = pipe(prompt, seed=0)
image.save("backdoor3_image_A_car_S*.jpg")

prompt = "A poster S*"
image = pipe(prompt, seed=0)
image.save("backdoor3_image_A_poster_S*.jpg")

prompt = "A white cat S*"
image = pipe(prompt, seed=0)
image.save("backdoor3_image_A_white_cat_S*.jpg")

prompt = "A car keychain S*"
image = pipe(prompt, seed=0)
image.save("backdoor3_image_A_car_keychain_S*.jpg")

prompt = "A car showroom with a poster, a car keychain S*"
image = pipe(prompt, seed=0)
image.save("backdoor3_image_A_car_showroom_with_a_poster,_a_car_keychain_S*.jpg")
