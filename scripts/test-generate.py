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
    prompt="Street level view from a cyberpunk city, concept art, high quality digital art, highly detailed, realistic, by michal lisowski, trending on artstation",  # noqa: E501
    num_inference_steps=50,
    seed=1234,
    nsfw_allowed=True,
)

init_img = Image.open("images/childs-drawing.jpg")
img2img_inputs = CombinedPipelineInputs(
    prompt="A family on a hike in Yosemite national park, concept art, 1800s oilpainting, realistic, muted colors, highly detailed",  # noqa: E501
    init_img=init_img,
    num_inference_steps=50,
    seed=1234,
    nsfw_allowed=True,
)

tshirt_inputs = CombinedPipelineInputs(
    prompt="A mexican skull design, calavera, large colorful design",
    num_inference_steps=50,
    seed=999,
    tshirt_mode=True,
)

pipe = CombinedPipeline()
pipe.to("cuda")
p = WorkerProcess(task_queue=None, slack_client=None)  # type: ignore

output_dir = Path(".outputs")
output_dir.mkdir(exist_ok=True)

text2img_results = p.generate(pipe, text2img_inputs)
for i, result in enumerate(text2img_results):
    result.img.save(output_dir / f"output-text2img-{i}.png")

img2img_results = p.generate(pipe, img2img_inputs)
for i, result in enumerate(img2img_results):
    result.img.save(output_dir / f"output-img2img-{i}.png")

tshirt_results = p.generate(pipe, tshirt_inputs)
for i, result in enumerate(tshirt_results):
    result.img.save(output_dir / f"output-tshirt-{i}.png")
