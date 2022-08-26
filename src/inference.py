import io
import logging
import multiprocessing as mp

import slack_sdk
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

logging.basicConfig(level=logging.INFO)


class Task:
    def __init__(self, prompt: str, channel: str):
        self.prompt = prompt
        self.channel = channel


class InferenceProcess(mp.Process):
    def __init__(self, task_queue: mp.Queue, slack_client: slack_sdk.WebClient):
        super().__init__()
        self.task_queue = task_queue
        self.slack_client = slack_client

    def load_model(self):
        # model_id = "CompVis/stable-diffusion-v1-4"
        model_id = "pipelines/sd-pipeline"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=True,
        )
        pipe = pipe.to("cuda")
        return pipe

    def generate(self, pipe, prompt: str) -> Image:
        with torch.autocast(pipe.device.type):
            results = pipe([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)
            img = results["sample"][0]
            return img

    def run(self):
        # Need to make sure to load the model in this forked process rather than
        # in the main process because otherwise CUDA complains.
        pipe = self.load_model()

        while True:
            task = self.task_queue.get()
            logging.info(f"Handling request: {task.prompt}")
            img = self.generate(pipe, task.prompt)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            self.slack_client.files_upload(
                channels=task.channel,
                title=task.prompt,
                content=img_bytes,
            )
