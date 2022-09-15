import io
import logging
import multiprocessing as mp

import slack_sdk
from PIL import Image
from pydantic import BaseModel

from .pipeline import CombinedPipeline
from .pipeline import CombinedPipelineInputs

logging.basicConfig(level=logging.INFO)

# TODO:
# - Fix missing type hints in this file.


class InferenceTask(BaseModel):
    inputs: CombinedPipelineInputs
    channel: str
    thread_ts: str


class InferenceResult(BaseModel):
    img: Image.Image
    nsfw: bool

    class Config:
        arbitrary_types_allowed = True


class InferenceProcess(mp.Process):
    def __init__(self, task_queue: mp.Queue, slack_client: slack_sdk.WebClient):
        super().__init__()
        self.task_queue = task_queue
        self.slack_client = slack_client

    def load_model(self) -> CombinedPipeline:
        pipe = CombinedPipeline("pipelines/sd-pipeline")
        pipe.to("cuda")
        return pipe

    def generate(self, pipe: CombinedPipeline, inputs: CombinedPipelineInputs) -> Image.Image:
        results = pipe(inputs)
        return [
            InferenceResult(img=img, nsfw=nsfw)
            for img, nsfw in zip(results["sample"], results["nsfw_content_detected"])
        ]

    def upload_image(self, img: Image.Image, channel: str, title: str):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        self.slack_client.files_upload(
            channels=channel,
            title=title,
            content=img_bytes,
        )

    def run(self):
        # Need to make sure to load the model in this forked process rather than
        # in the main process because otherwise CUDA complains.
        pipe = self.load_model()

        logging.info("Inference ready to handle requests")
        while True:
            task = self.task_queue.get()
            logging.info(f"Handling request: {task}")

            results = self.generate(pipe, task.inputs)

            if all(result.nsfw for result in results):
                self.slack_client.chat_postMessage(
                    text="Oops! I generated something NSFW! Total accident ( ͡° ͜ʖ ͡°)",
                    channel=task.channel,
                    thread_ts=task.thread_ts,
                )
            else:
                results = [result for result in results if not result.nsfw]
                for result in results:
                    self.upload_image(
                        img=result.img, channel=task.channel, title=task.inputs.prompt
                    )

                # Write a reply in original message with instructions for how to reproduce results.
                config = task.inputs.dict()
                prompt = config.pop("prompt")
                config_str = ", ".join(f"{k}={v}" for k, v in config.items() if v is not None)
                command_to_reproduce = f"Use this command to reproduce the same result:\n`@burgerman {config_str} | {prompt}`"  # noqa: E501
                self.slack_client.chat_postMessage(
                    text=command_to_reproduce,
                    channel=task.channel,
                    thread_ts=task.thread_ts,
                )
