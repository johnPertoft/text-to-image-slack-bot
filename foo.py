import time

import torch
from diffusers import StableDiffusionPipeline
from torch import autocast  # noqa: F401

# TODO: See https://huggingface.co/CompVis/stable-diffusion-v1-4
# TODO: Move model files to gs:// to avoid downloading them everytime.

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
torch_dtype = None
torch_dtype = torch.float16

# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
# pipe = pipe.to(device)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, use_auth_token=True, torch_dtype=torch_dtype
)
pipe = pipe.to(device)

prompt = "A black and white photo of a robot playing chess in the 1960s"
with autocast(device):
    t1 = time.time()
    results = pipe([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)
    t2 = time.time()

print(f"Elapsed time: {t2 - t1}")
images = results["sample"]

for i, img in enumerate(images):
    img.save(f"{i:03}.png")
