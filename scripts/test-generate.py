#!/usr/bin/python

"""
This script runs just the code to generate an image.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from PIL import Image  # noqa: E402

from src.inference import InferenceProcess  # noqa: E402
from src.pipeline import CombinedPipeline  # noqa: E402
from src.pipeline import CombinedPipelineInputs  # noqa: E402

pipe = CombinedPipeline("pipelines/sd-pipeline")
pipe.to("cuda")

init_img = Image.open("images/childs-drawing.jpg")

inputs = CombinedPipelineInputs(
    prompt="A small family on a hike in Yosemite national park",
    num_inference_steps=50,
    seed=1234,
    nsfw_allowed=True,
    init_img=init_img,
)
p = InferenceProcess(None, None)  # type: ignore
img, has_nsfw = p.generate(pipe, inputs)

img.save("output.png")
