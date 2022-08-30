from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers import PNDMScheduler
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from transformers import CLIPFeatureExtractor
from transformers import CLIPTextModel
from transformers import CLIPTokenizer

from .contrib import StableDiffusionImg2ImgPipeline  # type: ignore


def preprocess_img(image: Image.Image) -> torch.FloatTensor:
    # TODO: Keep aspect ratio and do cropping maybe.
    # TODO: Don't hardcode it here.
    image = image.resize((768, 512))

    # TODO: Need to ensure that size fits.
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class BurgermanPipeline:
    def __init__(self, pipeline_path: str):
        # TODO: Correct models being used? Blog post specified "openai/clip-vit-large-patch14"
        # for text encoder.

        # TODO: Fix this warning? ftfy or spacy is not installed using BERT BasicTokenizer
        # instead of ftfy.

        # TODO: Potentially replace the contrib pipeline since it barely does anything different
        # I think? Only replaces the initial latent from the input image.

        # TODO: Experiment with different schedulers. See blog post.

        # TODO: The two different pipeline have some different expectations on schedulers.
        # Read up on this.

        pipeline_dir = Path(pipeline_path)
        tokenizer = CLIPTokenizer.from_pretrained(pipeline_dir / "tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pipeline_dir / "text_encoder")
        vae = AutoencoderKL.from_pretrained(pipeline_dir / "vae")
        unet = UNet2DConditionModel.from_pretrained(pipeline_dir / "unet")
        scheduler = PNDMScheduler.from_config(pipeline_dir / "scheduler")

        # TODO: Could potentially patch out the nsfw filter here by just supplying functions
        # returning false?
        feature_extractor = CLIPFeatureExtractor.from_pretrained(pipeline_dir / "feature_extractor")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            pipeline_dir / "safety_checker"
        )

        self.text2img = StableDiffusionPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        self.img2img = StableDiffusionImg2ImgPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def to(self, device: str) -> None:
        self.text2img = self.text2img.to(device)
        self.img2img = self.img2img.to(device)

    @property
    def device(self):
        return self.text2img.device

    @torch.no_grad()
    def from_text(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        with torch.autocast(self.text2img.device.type):
            return self.text2img(
                prompt,
                height,
                width,
                num_inference_steps,
                guidance_scale,
                eta,
                generator,
                output_type,
                **kwargs,
            )

    @torch.no_grad()
    def from_text_and_image(
        self,
        prompt: Union[str, List[str]],
        init_image: torch.FloatTensor,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
    ):
        with torch.autocast(self.img2img.device.type):
            return self.img2img(
                prompt,
                init_image,
                strength,
                num_inference_steps,
                guidance_scale,
                eta,
                generator,
                output_type,
            )
