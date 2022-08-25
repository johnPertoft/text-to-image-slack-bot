import io
import logging
import os
import re
from typing import Any
from typing import Callable
from typing import Dict

from slack_bolt import App
from slack_bolt import BoltRequest
from slack_bolt import Say

from .generate import ImageGenerator
from .utils import get_secret

logging.basicConfig(level=logging.DEBUG)


logging.info("Loading model")
image_generator = ImageGenerator()

app = App(
    token=get_secret("john-test-slack-bot-token"),
    signing_secret=get_secret("john-test-slack-signing-secret"),
)


@app.middleware
def skip_retries(request: BoltRequest, next: Callable):
    if "X-Slack-Retry-Num" in request.headers:
        return
    next()


# TODO: There seems to be some ghost requests that eat up some requests.
# TODO: Should be a counter, and if running on multiple instances, need to sync somehow.
# from collections import defaultdict
# num_requests = defaultdict(int)


# @app.middleware
# def rate_limit(request: BoltRequest, next: Callable):
#     # TODO: Can we respond something in here too? To let the user know.
#     user_id = request.body["event"]["user"]

#     if num_requests[user_id] > 3:
#         print("========== SKIPPED!")
#         return
#     num_requests[user_id] += 1
#     next()


@app.event("app_mention", middleware=[skip_retries])
def app_mention(body: Dict[str, Any], say: Say, logger: logging.Logger):
    logger.info(body)
    raw_event_text = body["event"]["text"]
    event_text_match = re.search(r"(<@.*>) (.*)", raw_event_text)
    if event_text_match is not None:
        prompt = event_text_match.group(2)
    else:
        prompt = "default prompt"

    say(f"Hey! I'll generate an image for: {prompt}")
    img = image_generator.generate(prompt)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    app.client.files_upload(
        channels=say.channel,
        title=prompt,
        content=img_bytes,
    )


if __name__ == "__main__":
    port = os.environ.get("PORT", 3000)
    port = int(port)
    app.start(port=port)
