import contextlib
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .query import Query


class CombinedPipelineInputs(Query):
    seed: int
    init_img: Optional[Image.Image]

    class Config:
        arbitrary_types_allowed = True


def preprocess_img(image: Image.Image) -> torch.FloatTensor:
    # TODO:
    # - The collab was running with a t4 too, why wasn't it running
    #   out of mem with 512x1024 img?
    # - Do resizing + cropping properly instead.
    # - Needs to be a multiple of 32. Add assertion.
    w, h = image.size
    if w / h >= 1.3:
        image = image.resize((768, 512), resample=Image.LANCZOS)
    elif h / w >= 1.3:
        image = image.resize((512, 768), resample=Image.LANCZOS)
    else:
        image = image.resize((512, 512), resample=Image.LANCZOS)

    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


# TODO: Remove this? And the nsfw_allowed option.
@contextlib.contextmanager
def maybe_bypass_nsfw(pipe, nsfw_allowed: bool):
    def dummy_safety_checker(images, *args, **kwargs):
        has_nsfw_concept = [False] * len(images)
        return images, has_nsfw_concept

    original_safety_checker = pipe.safety_checker
    if nsfw_allowed:
        pipe.safety_checker = dummy_safety_checker
    yield pipe
    pipe.safety_checker = original_safety_checker


class CombinedPipeline:
    def __init__(self, pipeline_path: str):
        # It seems like diffusers 0.4^ loads cuda on import which breaks the setup here
        # with cuda being loaded in a separate process from the main app process.
        # So instead we delay the import and have it here instead.
        from diffusers import EulerDiscreteScheduler
        from diffusers import StableDiffusionImg2ImgPipeline
        from diffusers import StableDiffusionInpaintPipelineLegacy
        from diffusers import StableDiffusionPipeline

        pipeline_dir = Path(pipeline_path)

        # TODO: Is there a benefit in using different schedulers for different pipelines?
        scheduler = EulerDiscreteScheduler.from_pretrained(pipeline_dir / "scheduler")

        self.text2img = StableDiffusionPipeline.from_pretrained(
            pipeline_dir, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )

        self.img2img = StableDiffusionImg2ImgPipeline(
            tokenizer=self.text2img.tokenizer,
            text_encoder=self.text2img.text_encoder,
            vae=self.text2img.vae,
            unet=self.text2img.unet,
            scheduler=self.text2img.scheduler,
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None,
        )

        # TODO:
        # - The non legacy one requires different weights/config for unet.
        # - Maybe we can load a separate unet for this one instead?
        self.inpainting = StableDiffusionInpaintPipelineLegacy(
            tokenizer=self.text2img.tokenizer,
            text_encoder=self.text2img.text_encoder,
            vae=self.text2img.vae,
            unet=self.text2img.unet,
            scheduler=self.text2img.scheduler,
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None,
        )

        # TODO: Increase the size of these?
        self.tshirt_img = Image.open("images/tshirt.jpeg").convert("RGB")
        self.tshirt_mask = Image.open("images/tshirt-mask.jpeg").convert("RGB")

    def to(self, device: str) -> None:
        self.text2img = self.text2img.to(device)
        self.img2img = self.img2img.to(device)

    @property
    def device(self):
        return self.text2img.device

    @torch.no_grad()
    def __call__(self, inputs: CombinedPipelineInputs) -> Dict[str, Any]:
        if inputs.tshirt_mode:
            return self.call_tshirt(inputs)
        elif inputs.init_img is not None:
            return self.call_img2img(inputs)
        else:
            return self.call_txt2img(inputs)

    def call_txt2img(self, inputs: CombinedPipelineInputs) -> Dict[str, Any]:
        random_generator = torch.Generator(self.device.type).manual_seed(inputs.seed)

        if inputs.format == "square":
            height = 768
            width = 768
        elif inputs.format == "wide":
            height = 768
            width = 1024
        else:
            height = 1024
            width = 768

        return self.text2img(
            prompt=inputs.prompt,
            height=height,
            width=width,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=inputs.guidance_scale,
            eta=0.0,
            generator=random_generator,
        )

    def call_img2img(self, inputs: CombinedPipelineInputs) -> Dict[str, Any]:
        random_generator = torch.Generator(self.device.type).manual_seed(inputs.seed)
        return self.img2img(
            prompt=inputs.prompt,
            image=inputs.init_img,
            strength=inputs.strength,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=inputs.guidance_scale,
            eta=0.0,
            generator=random_generator,
        )

    def call_tshirt(self, inputs: CombinedPipelineInputs) -> Dict[str, Any]:
        random_generator = torch.Generator(self.device.type).manual_seed(inputs.seed)
        return self.inpainting(
            prompt=inputs.prompt,
            image=self.tshirt_img,
            mask_image=self.tshirt_mask,
            strength=inputs.strength,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=inputs.guidance_scale,
            eta=0.0,
            generator=random_generator,
        )
