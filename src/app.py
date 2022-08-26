import logging
import multiprocessing as mp
import os
import re
from typing import Any
from typing import Callable
from typing import Dict

from slack_bolt import App
from slack_bolt import BoltRequest
from slack_bolt import Say

from .inference import InferenceProcess
from .inference import Task
from .utils import get_secret

logging.basicConfig(level=logging.INFO)

# Create the Slack app.
app = App(
    token=get_secret("john-test-slack-bot-token"),
    signing_secret=get_secret("john-test-slack-signing-secret"),
)

# TODO: I don't think this will work if running with gunicorn or some other web server.
# Create and start the inference process.
task_queue: mp.Queue = mp.Queue()
inference_process = InferenceProcess(task_queue, app.client)
inference_process.start()


@app.middleware
def skip_retries(request: BoltRequest, next: Callable):
    if "X-Slack-Retry-Num" in request.headers:
        return
    next()


@app.event("app_mention", middleware=[skip_retries])
def app_mention(body: Dict[str, Any], say: Say, logger: logging.Logger):
    raw_event_text = body["event"]["text"]
    event_text_match = re.search(r"(<@.*>) (.*)", raw_event_text)
    if event_text_match is not None:
        prompt = event_text_match.group(2)
    else:
        prompt = "default prompt"

    say(f"Hey! I'll generate an image for: {prompt}")
    task_queue.put(Task(prompt=prompt, channel=say.channel))


if __name__ == "__main__":
    port = os.environ.get("PORT", 3000)
    port = int(port)
    app.start(port=port)
