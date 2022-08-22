import numpy as np
from PIL import Image


def generate(prompt: str) -> Image:
    print(prompt)

    x = np.random.randint(low=0, high=256, size=(512, 512, 3))
    x = x.astype(np.uint8)

    return Image.fromarray(x)
