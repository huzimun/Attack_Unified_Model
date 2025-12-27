from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/data1/humw/Pretrains"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:3",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
input_image = pipe(prompt=prompt, seed=0, num_inference_steps=40, height=1328, width=1024)
input_image.save("text2image.jpg")

# prompt = "将裙子改为粉色"
prompt = "少女手里举着牌子，牌子上写着“你好”"
# edit_image_auto_resize=True: auto resize input image to match the area of 1024*1024 with the original aspect ratio
image = pipe(prompt, edit_image=input_image, seed=1, num_inference_steps=40, height=1328, width=1024, edit_image_auto_resize=True)
image.save(f"text2image_edited1.jpg")

# edit_image_auto_resize=False: do not resize input image
image = pipe(prompt, edit_image=input_image, seed=1, num_inference_steps=40, height=1328, width=1024, edit_image_auto_resize=False)
image.save(f"text2image_edited2.jpg")
