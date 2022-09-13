import argparse
import logging
import multiprocessing as mp
import os
import random
from typing import Any
from typing import Callable
from typing import Dict

from pydantic import ValidationError
from slack_bolt import App
from slack_bolt import BoltRequest
from slack_bolt import Say

from .inference import InferenceProcess
from .inference import InferenceTask
from .pipeline import CombinedPipelineInputs
from .query import ParseQueryException
from .query import Query
from .query import parse_query
from .utils import DownloadError
from .utils import download_img
from .utils import get_secret

# TODO:
# - Improve help strings when something goes wrong.
# - If mention is "some text @burgerman some other text", are we
#   handling it correctly?
# - Convert to AsyncApp, since we're doing some downloading of images potentially.
# - Handle error at different stages better.
# - Restart inference process if it dies. Use multiprocessing.Pool with one worker instead?
# - Maybe utilize middleware to do error handling?
# - Can we generate usage string from pydantic object?

logging.basicConfig(level=logging.INFO)

USAGE_STR = """
Usage examples
@burgerman A horse in space
@burgerman seed=123, format=wide | A horse in space
@burgerman img_uri=https://url.to.my/image.png | A horse in space

Config options
seed: int
img_uri: HttpUrl, use this image as starting image
num_inference_steps: int: [1, 100], default 50
guidance_scale: float: [1.0, 15.0], default 7.5
strength: float: [0.0, 1.0], default 0.8, only used for img2img
format: Literal["square", "tall", "wide"]
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
app = App(
    token=get_secret("john-test-slack-bot-token"),
    signing_secret=get_secret("john-test-slack-signing-secret"),
    request_verification_enabled=not args.skip_request_verification,
)

# Create and start the inference process.
task_queue: mp.Queue = mp.Queue()
inference_process = InferenceProcess(task_queue, app.client)
inference_process.start()


def prepare_pipeline_inputs(query: Query) -> CombinedPipelineInputs:
    if query.img_uri is not None:
        logging.info("Downloading {query.img_uri}")
        img = download_img(query.img_uri, slack_token=get_secret("john-test-slack-bot-token"))
    else:
        img = None

    return CombinedPipelineInputs(
        prompt=query.prompt,
        seed=query.seed or random.randint(0, 10000),
        guidance_scale=query.guidance_scale,
        num_inference_steps=query.num_inference_steps,
        img_uri=query.img_uri,
        init_img=img,
        format=query.format,
        nsfw_allowed=query.nsfw_allowed,
    )


@app.middleware
def skip_retries(request: BoltRequest, next: Callable):
    if "X-Slack-Retry-Num" in request.headers:
        return
    return next()


@app.event("app_mention", middleware=[skip_retries])
def app_mention(body: Dict[str, Any], say: Say):
    thread_ts = body["event"]["ts"]

    def say_in_thread(msg: str):
        say(msg, thread_ts=thread_ts)

    raw_event_text = body["event"]["text"]
    try:
        query = parse_query(raw_event_text)
    except ParseQueryException as e:
        say_in_thread(f"Oops! {e}\n\n{USAGE_STR}")
        return
    except ValidationError:
        say_in_thread(f"Oops! I couldn't validate those inputs!\n\n{USAGE_STR}")
        return

    try:
        pipeline_inputs = prepare_pipeline_inputs(query)
    except DownloadError as e:
        say_in_thread(f"Oops! {e}")
        return
    except ValidationError:
        say_in_thread(f"Oops! I couldn't validate those inputs!\n\n{USAGE_STR}")
        return

    # Acknowledge that the query was parsed correctly and queue up the inference request.
    app.client.reactions_add(
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
