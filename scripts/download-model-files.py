import torch
from diffusers import StableDiffusionPipeline

model_ids = [
    "stabilityai/stable-diffusion-2",
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
]
for model_id in model_ids:
    StableDiffusionPipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    ).save_pretrained(f"pipelines/{model_id}")
