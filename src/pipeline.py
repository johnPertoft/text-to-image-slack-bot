# TODO: Custom pipeline that can take just text or text+image.
# TODO: Probably can't fit two different pipelines so better to have one handling both.
# TODO: ...or maybe load the parts (vae, unet, etc) and then create two pipelines from them,
# easier to maintain?

import io

import numpy as np
import requests  # type: ignore
import torch
from PIL import Image


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


# TODO: Use this to load image url somewhere
# TODO: Maybe worth it to convert to async app since we will spend time on downloading now too.
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"  # noqa: E501
response = requests.get(url)
init_image = Image.open(io.BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
init_image = preprocess(init_image)
