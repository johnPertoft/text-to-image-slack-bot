#!/usr/bin/python
import torch
from diffusers import StableDiffusionPipeline

StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2", revision="fp16", torch_dtype=torch.float16
).save_pretrained("pipelines/stabilityai/stable-diffusion-2")
