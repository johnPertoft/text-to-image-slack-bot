import logging
import os
import re
import time
from typing import Any
from typing import Dict

from google.cloud import secretmanager
from slack_bolt import App
from slack_bolt import Say

logging.basicConfig(level=logging.DEBUG)


# TODO: AsyncApp instead.
# TODO: Add requests to a queue and handle in batches and respond later.
# TODO: Also see https://slack.dev/bolt-python/concepts#lazy-listeners
# Or not an issue when using say()? Doesn't seem to require an ack within
# 3 secs.
# TODO: Use socket mode?
# TODO: Where to deploy? Cloud run?
# TODO: Responding multiple times, because of no ack?


def _get_secret(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    full_secret_name = f"projects/embark-nlp/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=full_secret_name)
    return response.payload.data.decode("UTF-8")


app = App(
    token=_get_secret("john-test-slack-bot-token"),
    signing_secret=_get_secret("john-test-slack-signing-secret"),
)


@app.event("app_mention")
def app_mention(body: Dict[str, Any], say: Say, logger: logging.Logger):
    logger.info(body)

    raw_event_text = body["event"]["text"]
    event_text_match = re.search(r"(<@.*>) (.*)", raw_event_text)
    if event_text_match is not None:
        prompt = event_text_match.group(2)
    else:
        prompt = "default prompt"

    say(f"Hey! I'll generate an image for: {prompt}")

    # TODO: Generate the image.
    time.sleep(10)

    # TODO: This sends the image to the channel directly without an accompanying message.
    # TODO: Can we just upload and then send a link to it?
    # TODO: If not, we can set the channel to some other one with all of them.
    # TODO: Check response here.
    app.client.files_upload(
        channels=say.channel,
        file="img.png",
        title=prompt,
    )


if __name__ == "__main__":
    port = os.environ.get("PORT", 3000)
    port = int(port)
    app.start(port=port)
