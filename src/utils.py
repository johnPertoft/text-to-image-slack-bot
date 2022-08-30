import functools
import io
import logging
from typing import Optional

import requests  # type: ignore
from google.cloud import secretmanager
from PIL import Image

logging.basicConfig(level=logging.INFO)


def get_secret(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    full_secret_name = f"projects/embark-nlp/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=full_secret_name)
    return response.payload.data.decode("UTF-8")


@functools.lru_cache(maxsize=512)
def download_img(url: str) -> Optional[Image.Image]:
    # TODO:
    # - Should update to async server since we'll spend time downloading potentially.
    # - Stop if too big file or something?
    # - Stop if it's not an image
    # - General fault tolerance needed here.
    # - Catch only the expected errors here.
    # - Should maybe have some caching of image. I imagine that people might rerun with same url.
    # - Probably need to add some token to be able to download from slack.
    #   headers = {'Authorization': 'Bearer ' + TOKEN}
    #   requests.get(url, timeout=5.0, headers=headers)  Only for slack url though?
    try:
        response = requests.get(url, timeout=5.0)
        img_bytes = io.BytesIO(response.content)
        img = Image.open(img_bytes).convert("RGB")
        return img
    except Exception as e:
        # TODO: Catch only expected errors.
        logging.exception(f"Exception when downloading image: {e}")
        return None
