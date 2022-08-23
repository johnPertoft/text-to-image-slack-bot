import torch
from diffusers import StableDiffusionPipeline
from torch import autocast  # noqa: F401

# TODO: See https://huggingface.co/CompVis/stable-diffusion-v1-4
# TODO: Move model files to gs:// to avoid downloading them everytime.

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
# pipe = pipe.to(device)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, use_auth_token=True, torch_dtype=torch.float16
)
pipe = pipe.to(device)

prompt = "A painting of a squirrel eating a burger"
results = pipe([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)
images = results["sample"]

for i, img in enumerate(images):
    img.save(f"{i:03}.png")
