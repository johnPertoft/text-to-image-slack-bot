from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Extra
from pydantic import Field
from pydantic import HttpUrl

# TODO:
# - Have tests for this.
# - Just repeat some of these fields for the actual model inputs data class?
# - Do not accept additional fields to capture spelling errors
# - Post help message back to message thread.


class Query(BaseModel):
    prompt: str
    seed: Optional[int]
    img_uri: Optional[HttpUrl]
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=15.0)
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    format: Literal["square", "tall", "wide"] = "square"
    nsfw_allowed: bool = False

    class Config:
        extra = Extra.forbid


def parse_query(raw: str) -> Query:

    pass
