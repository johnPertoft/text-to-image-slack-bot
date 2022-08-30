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


def download_img(url: str) -> Optional[Image.Image]:
    # TODO:
    # - Should update to async server since we'll spend time downloading potentially.
    # - Stop if too big file or something?
    # - Stop if it's not an image
    # - General fault tolerance needed here.
    try:
        response = requests.get(url, timeout=5.0)
        img_bytes = io.BytesIO(response.content)
        img = Image.open(img_bytes).convert("RGB")
        return img
    except Exception as e:
        # TODO: Catch only expected errors.
        logging.exception(f"Exception when downloading image: {e}")
        return None
