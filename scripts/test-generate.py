"""
This script runs just the code to generate an image.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from PIL import Image  # noqa: E402

from src.pipeline import CombinedPipeline  # noqa: E402
from src.pipeline import CombinedPipelineInputs  # noqa: E402
from src.worker import WorkerProcess  # noqa: E402

text2img_inputs = CombinedPipelineInputs(
    prompt="Street level view from a cyberpunk city, concept art, high quality digital art, by michal lisowski, trending on artstation",  # noqa: E501
    num_inference_steps=50,
    seed=1234,
    nsfw_allowed=True,
)

init_img = Image.open("images/childs-drawing.jpg")
img2img_inputs = CombinedPipelineInputs(
    prompt="A family on a hike in Yosemite national park, concept art, high quality digital art, by micahl lisowski, trending on artstation",  # noqa: E501
    init_img=init_img,
    num_inference_steps=50,
    seed=1234,
    nsfw_allowed=True,
)

pipe = CombinedPipeline("pipelines/stable-diffusion-v1-4")
pipe.to("cuda")
p = WorkerProcess(None, None)  # type: ignore

text2img_results = p.generate(pipe, text2img_inputs)
for i, result in enumerate(text2img_results):
    result.img.save(f"output-text2img-{i}.png")

img2img_results = p.generate(pipe, img2img_inputs)
for i, result in enumerate(img2img_results):
    result.img.save(f"output-img2img-{i}.png")
