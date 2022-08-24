import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class ImageGenerator:
    def __init__(self):
        # TODO: Add code to download model if not present I guess.
        # "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            "pipelines/sd-pipeline", use_auth_token=True, torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        self.pipe = pipe

    def generate(self, prompt: str) -> Image:
        with torch.autocast(self.pipe.device.type):
            results = self.pipe([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)
            img = results["sample"][0]
            return img
