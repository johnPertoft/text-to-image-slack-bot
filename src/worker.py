import asyncio
import io
import logging
import multiprocessing as mp
from typing import List

from PIL import Image
from pydantic import BaseModel
from slack_sdk.web.async_client import AsyncWebClient

from .constants import SLACK_APP_NAME
from .pipeline import CombinedPipeline
from .pipeline import CombinedPipelineInputs


class InferenceTask(BaseModel):
    inputs: CombinedPipelineInputs
    channel: str
    thread_ts: str


class InferenceResult(BaseModel):
    img: Image.Image
    nsfw: bool

    class Config:
        arbitrary_types_allowed = True


class WorkerProcess(mp.Process):
    def __init__(self, task_queue: mp.Queue, slack_client: AsyncWebClient):
        super().__init__()
        self.task_queue = task_queue
        self.slack_client = slack_client

    def run(self):
        asyncio.run(self.handle_requests_forever())

    async def handle_requests_forever(self) -> None:
        # Need to make sure to load the model in this forked process rather than
        # in the main process because otherwise CUDA complains.
        pipe = CombinedPipeline("pipelines/stabilityai/stable-diffusion-2")
        pipe.to("cuda")

        logging.info("Inference ready to handle requests")
        while True:
            task = self.task_queue.get()
            try:
                await self.handle_request(task, pipe)
            except Exception as e:
                logging.warning(f"Caught unhandled exception: {e}")

    async def handle_request(self, task: InferenceTask, pipe: CombinedPipeline) -> None:
        logging.info(f"Handling request: {task}")
        results = self.generate(pipe, task.inputs)

        # TODO: This can't happen anymore, remove?
        # Write a reply if all results were nsfw and early exit.
        if all(result.nsfw for result in results):
            nsfw_msg = "\n".join(
                [
                    "Oops! All results were NSFW!",
                    f"You can retry with `@{SLACK_APP_NAME} nsfw_allowed=True | {task.inputs.prompt}`",  # noqa: E501
                ]
            )
            return await self.slack_client.chat_postMessage(
                text=nsfw_msg,
                channel=task.channel,
                thread_ts=task.thread_ts,
            )

        # Upload images to Slack.
        results = [result for result in results if not result.nsfw]
        images = [r.img for r in results]
        await self.upload_images(images, channel=task.channel, title=task.inputs.prompt)

        # Write a reply in original message with instructions for how to reproduce results.
        config = task.inputs.dict()
        prompt = config.pop("prompt")
        config_strs = []
        for k, v in config.items():
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    config_strs.append(f"--{k}")
            else:
                config_strs.append(f"--{k}={v}")
        config_str = " ".join(config_strs)
        command_to_reproduce = "\n".join(
            [
                "Use this command to reproduce the same result:",
                f"`@{SLACK_APP_NAME} {prompt} {config_str}`",
            ]
        )
        await self.slack_client.chat_postMessage(
            text=command_to_reproduce,
            channel=task.channel,
            thread_ts=task.thread_ts,
        )

    async def upload_images(self, imgs: List[Image.Image], channel: str, title: str) -> None:
        img_uploads = []
        for i, img in enumerate(imgs):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            img_uploads.append({"title": title, "content": img_bytes, "filename": f"img-{i}.png"})

        return await self.slack_client.files_upload_v2(
            channel=channel,
            file_uploads=img_uploads,
        )

    def generate(self, pipe: CombinedPipeline, inputs: CombinedPipelineInputs) -> Image.Image:
        results = pipe(inputs)
        return [InferenceResult(img=img, nsfw=False) for img in results.images]
