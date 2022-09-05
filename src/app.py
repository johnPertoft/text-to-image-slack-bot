import argparse
import configparser
import logging
import multiprocessing as mp
import os
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from pydantic import ValidationError
from slack_bolt import App
from slack_bolt import BoltRequest
from slack_bolt import Say

from .errors import DownloadError
from .inference import InferenceInputs
from .inference import InferenceProcess
from .inference import InferenceTask
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

logging.basicConfig(level=logging.INFO)

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


def parse_query(query_msg: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    # Expected query msg format:
    # guidance_scale = 8.5, format = wide | An old red car, 1950s
    parts = query_msg.split("|")
    if len(parts) == 1:
        prompt_str = parts[0]
        prompt_str = prompt_str.strip()

        return prompt_str, None
    else:
        config_str = parts[0]
        prompt_str = "|".join(parts[1:])  # If "|" present in prompt part, join it back.
        prompt_str = prompt_str.strip()

        config_str = config_str.replace(" ", "")
        config_str = config_str.replace(",", "\n")
        config_str = f"[config]\n{config_str}"
        config = configparser.ConfigParser()
        config.read_string(config_str)
        config_d = dict(config["config"])

        return prompt_str, config_d


def prepare_inputs(prompt: str, config: Optional[Dict[str, Any]]) -> InferenceInputs:
    config = config or dict()
    logging.info(f"Preparing inputs from config: {config}")
    if "img_uri" in config:
        # TODO: Do this with regex.
        # Slack adds some formatting to urls so we need to remove it.
        img_uri = config["img_uri"]
        img_uri = img_uri[1:-1]
        logging.info(f"Downloading {img_uri}")
        img = download_img(img_uri, slack_token=get_secret("john-test-slack-bot-token"))
        config["init_img"] = img
        del config["img_uri"]
    return InferenceInputs(prompt=prompt, **config)


@app.middleware
def skip_retries(request: BoltRequest, next: Callable):
    if "X-Slack-Retry-Num" in request.headers:
        return
    return next()


@app.event("app_mention", middleware=[skip_retries])
def app_mention(body: Dict[str, Any], say: Say):
    thread_ts = body["event"]["ts"]

    def say_error(msg: str):
        say(msg, thread_ts=thread_ts)

    # Get the query from the @mention message.
    raw_event_text = body["event"]["text"]
    raw_event_query_match = re.search(r"(<@.*?>) (.*)", raw_event_text)
    if raw_event_query_match is None:
        say_error("Oops! You have to write a text prompt too!")
        return
    query_msg = raw_event_query_match.group(2)

    # Parse prompt and optional configuration.
    try:
        prompt, config = parse_query(query_msg)
    except configparser.ParsingError:
        say_error("Oops! I couldn't parse that configuration.")
        return

    # Prepare inputs.
    try:
        inference_inputs = prepare_inputs(prompt, config)
    except DownloadError:
        say_error("Oops! I couldn't download that image.")
        return
    except ValidationError:
        say_error("Oops! I couldn't validate those inputs.")
        return

    # Queue up the inference request.
    say(f"Hey! I'll generate an image for {inference_inputs.prompt}")
    task_queue.put(
        InferenceTask(
            inputs=inference_inputs,
            channel=say.channel,
            thread_ts=thread_ts,
            title=query_msg,
        )
    )


if __name__ == "__main__":
    port = os.environ.get("PORT", 3000)
    port = int(port)
    app.start(port=port)
