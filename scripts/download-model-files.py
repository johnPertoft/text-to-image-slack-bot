#!/usr/bin/python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
pipe.save_pretrained("pipelines/stable-diffusion-v1-4")
