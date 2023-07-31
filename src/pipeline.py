from typing import Any
from typing import Dict
from typing import Optional

import torch
from PIL import Image

from .query import Query


class CombinedPipelineInputs(Query):
    seed: int
    init_img: Optional[Image.Image]

    class Config:
        arbitrary_types_allowed = True


class CombinedPipeline:
    def __init__(self):
        # TODO: Check if this delayed import is still necessary
        # It seems like diffusers 0.4^ loads cuda on import which breaks the setup here
        # with cuda being loaded in a separate process from the main app process.
        # So instead we delay the import and have it here instead.
        from diffusers import StableDiffusionXLImg2ImgPipeline
        from diffusers import StableDiffusionXLInpaintPipeline
        from diffusers import StableDiffusionXLPipeline

        self.text2img = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        # TODO: What is this lora ckpt doing exactly?
        # self.text2img.load_lora_weights(
        #     "stabilityai/stable-diffusion-xl-base-1.0",
        #     weight_name="sd_xl_offset_example-lora_1.0.safetensors",
        # )
        self.text2img.to(torch_dtype=torch.float16)

        self.img2img = StableDiffusionXLImg2ImgPipeline(
            vae=self.text2img.vae,
            text_encoder=self.text2img.text_encoder,
            text_encoder_2=self.text2img.text_encoder_2,
            tokenizer=self.text2img.tokenizer,
            tokenizer_2=self.text2img.tokenizer_2,
            unet=self.text2img.unet,
            scheduler=self.text2img.scheduler,
        )

        self.inpainting = StableDiffusionXLInpaintPipeline(
            vae=self.text2img.vae,
            text_encoder=self.text2img.text_encoder,
            text_encoder_2=self.text2img.text_encoder_2,
            tokenizer=self.text2img.tokenizer,
            tokenizer_2=self.text2img.tokenizer_2,
            unet=self.text2img.unet,
            scheduler=self.text2img.scheduler,
        )

        self.tshirt_img = Image.open("images/tshirt.jpeg").convert("RGB")
        self.tshirt_mask = Image.open("images/tshirt-mask.jpeg").convert("RGB")

    def to(self, device: str) -> None:
        self.text2img.to(device)
        self.img2img.to(device)
        self.inpainting.to(device)

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
            height = 1024
            width = 1024
        elif inputs.format == "wide":
            height = 1024
            width = 1536
        else:
            height = 1536
            width = 1024

        return self.text2img(
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
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
            negative_prompt=inputs.negative_prompt,
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
            negative_prompt=inputs.negative_prompt,
            image=self.tshirt_img,
            mask_image=self.tshirt_mask,
            strength=inputs.strength,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=inputs.guidance_scale,
            eta=0.0,
            generator=random_generator,
        )
