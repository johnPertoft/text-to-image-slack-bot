import asyncio
import io
import multiprocessing as mp
from typing import List

from loguru import logger
from PIL import Image
from pydantic import BaseModel
from slack_sdk.web.async_client import AsyncSlackResponse
from slack_sdk.web.async_client import AsyncWebClient

from .constants import SLACK_APP_NAME
from .pipeline import CombinedPipeline
from .pipeline import CombinedPipelineInputs
from .query import get_flags_string


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
        pipe = CombinedPipeline()
        pipe.to("cuda")

        logger.info("Inference ready to handle requests")
        while True:
            task = self.task_queue.get()
            try:
                await self.handle_request(task, pipe)
            except Exception as e:
                logger.warning(f"Caught unhandled exception: {e}")

    async def handle_request(self, task: InferenceTask, pipe: CombinedPipeline) -> None:
        logger.info(f"Handling request: {task}")
        results = self.generate(pipe, task.inputs)

        # Upload images to Slack.
        results = [result for result in results if not result.nsfw]
        images = [r.img for r in results]
        await self.upload_images(images, channel=task.channel, title=task.inputs.prompt)

        # Write a reply in original message with instructions for how to reproduce results.
        config = task.inputs.dict()
        prompt = config.pop("prompt")
        config_str = get_flags_string(config)
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

    async def upload_images(
        self, imgs: List[Image.Image], channel: str, title: str
    ) -> AsyncSlackResponse:
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

    def generate(
        self, pipe: CombinedPipeline, inputs: CombinedPipelineInputs
    ) -> List[InferenceResult]:
        results = pipe(inputs)
        return [InferenceResult(img=img, nsfw=False) for img in results.images]
