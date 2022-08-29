import argparse
import configparser
import logging
import multiprocessing as mp
import os
import re
import shlex
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from pydantic import ValidationError
from slack_bolt import App
from slack_bolt import BoltRequest
from slack_bolt import Say

from .inference import InferenceInputs
from .inference import InferenceProcess
from .inference import InferenceTask
from .utils import get_secret

logging.basicConfig(level=logging.INFO)

# Create the Slack app.
app = App(
    token=get_secret("john-test-slack-bot-token"),
    signing_secret=get_secret("john-test-slack-signing-secret"),
)

# Create and start the inference process.
task_queue: mp.Queue = mp.Queue()
inference_process = InferenceProcess(task_queue, app.client)
inference_process.start()

# Create an arg parser for requests.
inference_argparser = argparse.ArgumentParser()
inference_argparser.add_argument("--num_inference_steps", type=int)
inference_argparser.add_argument("--guidance_scale", type=float)
inference_argparser.add_argument("--format", type=str)
inference_argparser.add_argument("--seed", type=int)
inference_argparser.add_argument("prompt_words", nargs="+")

# TODO: Maybe use some format like this instead? comma separated toml?
# @burgerman format: wide, num_inference_steps: 100 | Baby Yoda learning to ride a bike
# @burgerman format = wide, num_inference_steps = 100 | Baby Yoda learning to ride a bike

# use implicits if not passing, i.e. same functionality as before
# @burgerman Baby Yoda learning to ride a bike


def parse_inputs(query_msg: str) -> Optional[InferenceInputs]:
    try:
        args = inference_argparser.parse_args(shlex.split(query_msg))
    except SystemExit:
        # TODO: In python 3.9 we can pass exit_on_error=False instead.
        return None

    try:
        args_d = {k: v for k, v in vars(args).items() if v is not None}
        args_d["prompt"] = shlex.join(args_d["prompt_words"])
        del args_d["prompt_words"]
        return InferenceInputs(**args_d)
    except ValidationError:
        return None


def parse_inputs2(query_msg: str) -> Optional[InferenceInputs]:
    # Expected query msg format:
    # guidance_scale = 8.5, format = wide | An old red car, 1950s

    # TODO: Not getting any type converters with this approach though.

    parts = query_msg.split("|")
    if len(parts) == 1:
        prompt_str = parts[0]
        return InferenceInputs(prompt=prompt_str)
    else:
        config_str = parts[0]
        prompt_str = "|".join(parts[1:])

        config_str = config_str.replace(" ", "")
        config_str = config_str.replace(",", "\n")
        config_str = f"[config]\n{config_str}"
        config = configparser.ConfigParser()
        try:
            config.read_string(config_str)
            # TODO: Create the InferenceInputs here
            return None

        except configparser.ParsingError:
            return None


@app.middleware
def skip_retries(request: BoltRequest, next: Callable):
    if "X-Slack-Retry-Num" in request.headers:
        return
    next()


@app.event("app_mention", middleware=[skip_retries])
def app_mention(body: Dict[str, Any], say: Say):
    raw_event_text = body["event"]["text"]
    raw_event_query_match = re.search(r"(<@.*>) (.*)", raw_event_text)

    if raw_event_query_match is None:
        say("Hey! You have to write a text prompt too!")
        return

    query_msg = raw_event_query_match.group(2)
    inference_inputs = parse_inputs(query_msg)

    if inference_inputs is None:
        # TODO: Better help message.
        # TODO: Maybe write in a thread response instead.
        say("Hey! I couldn't parse that request properly!")
        return

    say(f"Hey! I'll generate an image for {inference_inputs.prompt}")
    task_queue.put(InferenceTask(inputs=inference_inputs, channel=say.channel))


if __name__ == "__main__":
    port = os.environ.get("PORT", 3000)
    port = int(port)
    app.start(port=port)
