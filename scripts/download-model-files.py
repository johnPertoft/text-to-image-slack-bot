#!/usr/bin/python
from diffusers import StableDiffusionPipeline

StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2").save_pretrained(
    "pipelines/stabilityai/stable-diffusion-2"
)
