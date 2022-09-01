import io
import logging
import multiprocessing as mp
from typing import Literal
from typing import Optional

import slack_sdk
import torch
from PIL import Image
from pydantic import BaseModel
from pydantic import Field

from .pipeline import BurgermanPipeline
from .pipeline import preprocess_img

logging.basicConfig(level=logging.INFO)

# TODO:
# - Fix missing type hints in this file.


class InferenceInputs(BaseModel):
    prompt: str
    seed: Optional[int]
    init_img: Optional[Image.Image]
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=15.0)
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    format: Literal["square", "tall", "wide"] = "square"
    nsfw_allowed: bool = False

    class Config:
        arbitrary_types_allowed = True


class InferenceTask(BaseModel):
    inputs: InferenceInputs
    channel: str
    title: str


class InferenceProcess(mp.Process):
    def __init__(self, task_queue: mp.Queue, slack_client: slack_sdk.WebClient):
        super().__init__()
        self.task_queue = task_queue
        self.slack_client = slack_client

    def load_model(self) -> BurgermanPipeline:
        pipe = BurgermanPipeline("pipelines/sd-pipeline")
        pipe.to("cuda")
        return pipe

    def generate(self, pipe: BurgermanPipeline, inputs: InferenceInputs) -> Image.Image:
        if inputs.seed is not None:
            random_generator = torch.Generator(pipe.device.type).manual_seed(inputs.seed)
        else:
            random_generator = None

        if inputs.init_img is None:
            if inputs.format == "square":
                height = 512
                width = 512
            elif inputs.format == "wide":
                height = 512
                width = 768
            else:
                height = 768
                width = 512

            results = pipe.from_text(
                prompt=inputs.prompt,
                num_inference_steps=inputs.num_inference_steps,
                guidance_scale=inputs.guidance_scale,
                height=height,
                width=width,
                generator=random_generator,
                nsfw_allowed=inputs.nsfw_allowed,
            )
        else:
            init_img = preprocess_img(inputs.init_img)
            results = pipe.from_text_and_image(
                prompt=inputs.prompt,
                init_image=init_img,
                num_inference_steps=inputs.num_inference_steps,
                guidance_scale=inputs.guidance_scale,
                strength=inputs.strength,
                generator=random_generator,
                nsfw_allowed=inputs.nsfw_allowed,
            )

        img = results["sample"][0]
        nsfw_content_detected = results["nsfw_content_detected"]
        return img, nsfw_content_detected

    def run(self):
        # Need to make sure to load the model in this forked process rather than
        # in the main process because otherwise CUDA complains.
        pipe = self.load_model()

        logging.info("Inference ready to handle requests")
        while True:
            task = self.task_queue.get()
            logging.info(f"Handling request: {task}")
            img, nsfw_detected = self.generate(pipe, task.inputs)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            # TODO: Write in thread if nsfw was detected and skip upload.
            if nsfw_detected:
                logging.info("Detected NSFW output")
            self.slack_client.files_upload(
                channels=task.channel,
                title=task.title,
                content=img_bytes,
            )
