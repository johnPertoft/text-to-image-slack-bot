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
        img = results["sample"][0]
        nsfw_content_detected = results["nsfw_content_detected"][0]
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
            if nsfw_detected:
                self.slack_client.chat_postMessage(
                    text="Oops! I generated something NSFW",
                    channel=task.channel,
                    thread_ts=task.thread_ts,
                )
            else:
                config = task.inputs.dict()
                prompt = config.pop("prompt")
                config_str = ", ".join(f"{k}={v}" for k, v in config.items() if v is not None)
                command_to_reproduce = f"Use this command to reproduce the same result:\n`@burgerman {config_str} | {prompt}`"  # noqa: E501

                self.slack_client.files_upload(
                    channels=task.channel,
                    title=task.inputs.prompt,
                    content=img_bytes,
                )

                # TODO: Should post to the message with the image? Now it's the query message. Hmm.
                self.slack_client.chat_postMessage(
                    text=command_to_reproduce,
                    channel=task.channel,
                    thread_ts=task.thread_ts,
                )
