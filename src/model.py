import numpy as np
from PIL import Image

# TODO: Use diffusers library

"""
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)
"""


def generate(prompt: str) -> Image:
    print(prompt)

    x = np.random.randint(low=0, high=256, size=(512, 512, 3))
    x = x.astype(np.uint8)

    return Image.fromarray(x)
