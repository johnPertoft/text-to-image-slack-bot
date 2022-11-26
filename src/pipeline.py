import contextlib
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPFeatureExtractor
from transformers import CLIPTextModel
from transformers import CLIPTokenizer

from .query import Query

# TODO:
# - Just reimplement the pipeline logic
#   - Can have a single instead of separate branches for text2img and img2img
#   - Easier to skip the nsfw filter etc
# - Experiment with different schedulers, euler seems popular
# - There is a finetuned inpainting model too
#   https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion
# - Can we allow users to upload an image in slack to trigger img2img instead?


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
        from diffusers import AutoencoderKL
        from diffusers import PNDMScheduler
        from diffusers import StableDiffusionImg2ImgPipeline
        from diffusers import StableDiffusionInpaintPipelineLegacy
        from diffusers import StableDiffusionPipeline
        from diffusers import UNet2DConditionModel
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

        pipeline_dir = Path(pipeline_path)
        tokenizer = CLIPTokenizer.from_pretrained(pipeline_dir / "tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pipeline_dir / "text_encoder")
        vae = AutoencoderKL.from_pretrained(pipeline_dir / "vae")
        unet = UNet2DConditionModel.from_pretrained(pipeline_dir / "unet")
        scheduler = PNDMScheduler.from_config(pipeline_dir / "scheduler")
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

        # TODO: The non legacy one requires different weights/config for unet.
        self.inpainting = StableDiffusionInpaintPipelineLegacy(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

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
        pipe = self.text2img

        if inputs.format == "square":
            height = 512
            width = 512
        elif inputs.format == "wide":
            height = 512
            width = 768
        else:
            height = 768
            width = 512

        if height * width <= 512 * 512:
            # This image size allows us to run with batch size 2.
            prompt = [inputs.prompt] * 2
        else:
            prompt = [inputs.prompt]

        with maybe_bypass_nsfw(pipe, inputs.nsfw_allowed):
            with torch.autocast(pipe.device.type):
                return pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=inputs.num_inference_steps,
                    guidance_scale=inputs.guidance_scale,
                    eta=0.0,
                    generator=random_generator,
                )

    def call_img2img(self, inputs: CombinedPipelineInputs) -> Dict[str, Any]:
        random_generator = torch.Generator(self.device.type).manual_seed(inputs.seed)
        pipe = self.img2img
        init_img = preprocess_img(inputs.init_img)
        with maybe_bypass_nsfw(pipe, inputs.nsfw_allowed):
            with torch.autocast(pipe.device.type):
                return pipe(
                    prompt=inputs.prompt,
                    init_image=init_img,
                    strength=inputs.strength,
                    num_inference_steps=inputs.num_inference_steps,
                    guidance_scale=inputs.guidance_scale,
                    eta=0.0,
                    generator=random_generator,
                )

    def call_tshirt(self, inputs: CombinedPipelineInputs) -> Dict[str, Any]:
        random_generator = torch.Generator(self.device.type).manual_seed(inputs.seed)
        pipe = self.inpainting
        with maybe_bypass_nsfw(pipe, inputs.nsfw_allowed):
            with torch.autocast(pipe.device.type):
                return pipe(
                    prompt=inputs.prompt,
                    init_image=self.tshirt_img,
                    mask_image=self.tshirt_mask,
                    strength=inputs.strength,
                    num_inference_steps=inputs.num_inference_steps,
                    guidance_scale=inputs.guidance_scale,
                    eta=0.0,
                    generator=random_generator,
                )
