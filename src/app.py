import configparser
import logging
import multiprocessing as mp
import os
import re
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


def parse_inputs(query_msg: str) -> Optional[InferenceInputs]:
    # Expected query msg format:
    # guidance_scale = 8.5, format = wide | An old red car, 1950s
    parts = query_msg.split("|")
    if len(parts) == 1:
        prompt_str = parts[0]
        return InferenceInputs(prompt=prompt_str)
    else:
        config_str = parts[0]
        prompt_str = "|".join(parts[1:])  # If "|" present in prompt part, join it back.

        config_str = config_str.replace(" ", "")
        config_str = config_str.replace(",", "\n")
        config_str = f"[config]\n{config_str}"
        config = configparser.ConfigParser()
        try:
            config.read_string(config_str)
            return InferenceInputs(**dict(config["config"]), prompt=prompt_str)
        except configparser.ParsingError:
            return None
        except ValidationError:
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
        # TODO: Write this as a thread response instead + help string.
        say("Hey! You have to write a text prompt too!")
        return

    query_msg = raw_event_query_match.group(2)
    inference_inputs = parse_inputs(query_msg)

    if inference_inputs is None:
        # TODO: Write this as a thread response instead + help string.
        # TODO: Show if it was validation error?
        say("Hey! I couldn't parse that request properly!")
        return

    say(f"Hey! I'll generate an image for {inference_inputs.prompt}")
    task_queue.put(InferenceTask(inputs=inference_inputs, channel=say.channel))


if __name__ == "__main__":
    port = os.environ.get("PORT", 3000)
    port = int(port)
    app.start(port=port)
