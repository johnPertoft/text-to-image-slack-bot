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

pipe = CombinedPipeline("pipelines/stable-diffusion-v1-4")
pipe.to("cuda")

init_img = Image.open("images/childs-drawing.jpg")

inputs = CombinedPipelineInputs(
    # prompt="A small family on a hike in Yosemite national park",
    # init_img=init_img,
    prompt="Street level view from a cyberpunk city, concept art, high quality digital art, by michal lisowski, trending on artstation",  # noqa: E501
    num_inference_steps=50,
    seed=1234,
    nsfw_allowed=True,
)
p = InferenceProcess(None, None)  # type: ignore
results = p.generate(pipe, inputs)

for i, result in enumerate(results):
    result.img.save(f"output-{i}.png")
