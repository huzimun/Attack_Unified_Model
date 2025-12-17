from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/data1/humw/Pretrains"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:2",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
# prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
prompt = "A car showroom with a poster, a car keychain."
image = pipe(prompt, seed=0, num_inference_steps=30,height=512, width=512)
image.save("test_512.jpg")
# image = pipe(prompt, seed=10, num_inference_steps=30)
# image.save("image_en_car_seed-10.jpg")
# image = pipe(prompt, seed=20, num_inference_steps=30)
# image.save("image_en_car_seed-20.jpg")
# image = pipe(prompt, seed=30, num_inference_steps=30)
# image.save("image_en_car_seed-30.jpg")
# image = pipe(prompt, seed=40, num_inference_steps=30)
# image.save("image_en_car_seed-40.jpg")
# # prompt = "一个汽车展厅的海报上写着“豪华”，一个标有“驾驶”的汽车钥匙扣。"
# # image = pipe(prompt, seed=0, num_inference_steps=40)
# # image.save("image_zh_text.jpg")
