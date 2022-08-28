import io
import logging
import multiprocessing as mp
from typing import Literal
from typing import Optional

import slack_sdk
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from pydantic import BaseModel
from pydantic import Field

logging.basicConfig(level=logging.INFO)


class InferenceInputs(BaseModel):
    prompt: str
    seed: Optional[int]
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=15.0)
    format: Literal["square", "tall", "wide"] = "square"


class InferenceTask(BaseModel):
    inputs: InferenceInputs
    channel: str


class InferenceProcess(mp.Process):
    def __init__(self, task_queue: mp.Queue, slack_client: slack_sdk.WebClient):
        super().__init__()
        self.task_queue = task_queue
        self.slack_client = slack_client

    def load_model(self):
        model_id = "pipelines/sd-pipeline"
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to("cuda")
        return pipe

    def generate(self, pipe: StableDiffusionPipeline, inputs: InferenceInputs) -> Image:
        if inputs.seed is not None:
            random_generator = torch.Generator(pipe.device.type).manual_seed(inputs.seed)
        else:
            random_generator = None

        if inputs.format == "square":
            height = 512
            width = 512
        elif inputs.format == "wide":
            height = 512
            width = 768
        else:
            height = 768
            width = 512

        with torch.autocast(pipe.device.type):
            results = pipe(
                prompt=[inputs.prompt],
                num_inference_steps=inputs.num_inference_steps,
                guidance_scale=inputs.guidance_scale,
                height=height,
                width=width,
                generator=random_generator,
            )

        img = results["sample"][0]
        return img

    def run(self):
        # Need to make sure to load the model in this forked process rather than
        # in the main process because otherwise CUDA complains.
        pipe = self.load_model()

        logging.info("Inference ready to handle requests")
        while True:
            task = self.task_queue.get()
            logging.info(f"Handling request: {task}")
            img = self.generate(pipe, task.inputs)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            self.slack_client.files_upload(
                channels=task.channel,
                title=task.inputs.prompt,
                content=img_bytes,
            )
