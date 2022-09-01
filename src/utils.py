import functools
import io
import logging
from typing import Optional
from urllib.parse import urlparse

import requests  # type: ignore
from google.cloud import secretmanager
from PIL import Image

from .errors import DownloadError

logging.basicConfig(level=logging.INFO)


def get_secret(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    full_secret_name = f"projects/embark-nlp/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=full_secret_name)
    return response.payload.data.decode("UTF-8")


@functools.lru_cache(maxsize=512)
def download_img(url: str, slack_token: Optional[str] = None) -> Optional[Image.Image]:
    p = urlparse(url)
    if p.netloc == "files.slack.com" and slack_token is not None:
        headers = {"Authorization": "Bearer " + slack_token}
    else:
        headers = None
    try:
        response = requests.get(url, timeout=5.0, headers=headers)
        img_bytes = io.BytesIO(response.content)
        img = Image.open(img_bytes).convert("RGB")
        return img
    except Exception as e:
        # TODO: Catch only expected errors.
        logging.exception(f"Exception when downloading image: {e}")
        raise DownloadError()
