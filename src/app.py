import argparse
import logging
import multiprocessing as mp
import os
import random
from typing import Any
from typing import Callable
from typing import Dict

from pydantic import ValidationError
from slack_bolt import BoltRequest
from slack_bolt import Say
from slack_bolt.async_app import AsyncApp

from .constants import GCP_SLACK_BOT_TOKEN_SECRET_NAME
from .constants import GCP_SLACK_SIGNING_SECRET_NAME
from .constants import SLACK_APP_NAME
from .pipeline import CombinedPipelineInputs
from .query import ParseQueryException
from .query import Query
from .query import parse_query
from .utils import DownloadError
from .utils import download_img
from .utils import get_secret
from .worker import InferenceTask
from .worker import WorkerProcess

logging.basicConfig(level=logging.INFO)

USAGE_STR = f"""
Usage examples
@{SLACK_APP_NAME} A horse in space
@{SLACK_APP_NAME} seed=123, format=wide | A horse in space
@{SLACK_APP_NAME} img_url=https://url.to.my/image.png | A horse in space

Config options
seed: int
num_inference_steps: int: [1, 100], default 50
guidance_scale: float: [1.0, 15.0], default 7.5
img_url: HttpUrl, use this image as starting image in img2img
strength: float: [0.0, 1.0], default 0.8, only used for img2img
format: Literal["square", "tall", "wide"] default square, not used in img2img
"""

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--skip_request_verification",
    default=False,
    action="store_true",
    help="Whether request verification should be skipped. Useful for local testing.",
)
args = argparser.parse_args()

if args.skip_request_verification:
    logging.warning(
        "Running app without request verification! This should not be done in production!"
    )

# Create the Slack app.
app = AsyncApp(
    token=get_secret(GCP_SLACK_BOT_TOKEN_SECRET_NAME),
    signing_secret=get_secret(GCP_SLACK_SIGNING_SECRET_NAME),
    request_verification_enabled=not args.skip_request_verification,
)

# Create and start the inference worker process.
task_queue: mp.Queue = mp.Queue()
worker_process = WorkerProcess(task_queue, app.client)
worker_process.start()


async def prepare_pipeline_inputs(query: Query) -> CombinedPipelineInputs:
    if query.img_url is not None:
        logging.info(f"Downloading {query.img_url}")
        img = await download_img(
            query.img_url, slack_token=get_secret(GCP_SLACK_BOT_TOKEN_SECRET_NAME)
        )
    else:
        img = None

    return CombinedPipelineInputs(
        prompt=query.prompt,
        negative_prompt=query.negative_prompt,
        seed=query.seed or random.randint(0, 10000),
        guidance_scale=query.guidance_scale,
        num_inference_steps=query.num_inference_steps,
        img_url=query.img_url,
        init_img=img,
        format=query.format,
        nsfw_allowed=query.nsfw_allowed,
        tshirt_mode=query.tshirt_mode,
    )


@app.middleware
async def skip_retries(request: BoltRequest, next: Callable):
    if "X-Slack-Retry-Num" in request.headers:
        return
    return await next()


@app.event("app_mention", middleware=[skip_retries])
async def app_mention(body: Dict[str, Any], say: Say):
    thread_ts = body["event"]["ts"]

    async def say_in_thread(msg: str):
        return await say(msg, thread_ts=thread_ts)

    raw_event_text = body["event"]["text"]
    try:
        query = parse_query(raw_event_text)
    except ParseQueryException as e:
        return await say_in_thread(f"Oops! {e}\n\n{USAGE_STR}")
    except ValidationError:
        return await say_in_thread(f"Oops! I couldn't validate those inputs!\n\n{USAGE_STR}")

    try:
        pipeline_inputs = await prepare_pipeline_inputs(query)
    except DownloadError as e:
        return await say_in_thread(f"Oops! {e}")
    except ValidationError:
        return await say_in_thread(f"Oops! I couldn't validate those inputs!\n\n{USAGE_STR}")

    # Acknowledge that the query was parsed correctly and queue up the inference request.
    await app.client.reactions_add(
        name="ok",
        channel=say.channel,
        timestamp=thread_ts,
    )
    task_queue.put(
        InferenceTask(
            inputs=pipeline_inputs,
            channel=say.channel,
            thread_ts=thread_ts,
        )
    )


if __name__ == "__main__":
    port = os.environ.get("PORT", 3000)
    port = int(port)
    app.start(port=port)
