import logging
import os
import re
import tempfile
from typing import Any
from typing import Callable
from typing import Dict

import torch
from diffusers import StableDiffusionPipeline
from google.cloud import secretmanager
from PIL import Image
from slack_bolt import App
from slack_bolt import BoltRequest
from slack_bolt import Say
from torch import autocast

logging.basicConfig(level=logging.DEBUG)


# TODO:
# - AsyncApp instead?
# - Add requests to a queue and handle in batches and respond later.
# - Also see https://slack.dev/bolt-python/concepts#lazy-listeners
#   Or not an issue when using say()? Doesn't seem to require an ack within
#   3 secs.
# - Use socket mode?
# - Where to deploy? Cloud run? Need gke cluster with gpu nodes?
# - Responding multiple times, because of no ack? Fixed with middleware?


def _get_secret(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    full_secret_name = f"projects/embark-nlp/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=full_secret_name)
    return response.payload.data.decode("UTF-8")


model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
dtype = torch.float16
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True, torch_dtype=dtype)
pipe = pipe.to(device)


def _generate(prompt: str) -> Image:
    with autocast(device):
        results = pipe([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)
        img = results["sample"][0]
        return img


app = App(
    token=_get_secret("john-test-slack-bot-token"),
    signing_secret=_get_secret("john-test-slack-signing-secret"),
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

    img = _generate(prompt)
    img_path = tempfile.mktemp(suffix=".jpg")
    img.save(img_path)

    # TODO: This sends the image to the channel directly without an accompanying message.
    # TODO: Can we just upload and then send a link to it?
    # TODO: If not, we can set the channel to some other one with all of them.
    # TODO: Check response here.
    app.client.files_upload(
        channels=say.channel,
        file=img_path,
        title=prompt,
    )


if __name__ == "__main__":
    port = os.environ.get("PORT", 3000)
    port = int(port)
    app.start(port=port)
