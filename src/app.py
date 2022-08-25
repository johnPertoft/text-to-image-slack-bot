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
