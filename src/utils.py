import functools
import io
import logging
from typing import Optional
from urllib.parse import urlparse

import aiohttp
from google.cloud import secretmanager
from PIL import Image

logging.basicConfig(level=logging.INFO)


class DownloadError(Exception):
    pass


@functools.lru_cache()
def get_secret(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    full_secret_name = f"projects/embark-nlp/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=full_secret_name)
    return response.payload.data.decode("UTF-8")


# TODO: This cache doesn't work with async functions
# @functools.lru_cache(maxsize=512)
async def download_img(url: str, slack_token: Optional[str] = None) -> Image.Image:
    p = urlparse(url)
    if p.netloc == "files.slack.com" and slack_token is not None:
        headers = {"Authorization": "Bearer " + slack_token}
    else:
        headers = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5.0) as response:
                img_bytes = io.BytesIO(await response.read())
                img = Image.open(img_bytes).convert("RGB")
                return img
    except Exception as e:
        logging.exception(f"Exception when downloading image: {e}")
        raise DownloadError("I couldn't download that image!")
