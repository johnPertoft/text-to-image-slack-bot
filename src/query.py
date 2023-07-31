import argparse
import re
import shlex
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Extra
from pydantic import Field
from pydantic import HttpUrl
from pydantic import ValidationError

from .constants import SLACK_APP_NAME


class ParseQueryException(Exception):
    pass


class Query(BaseModel):
    prompt: str
    negative_prompt: Optional[str]
    seed: Optional[int]
    img_url: Optional[HttpUrl]
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=15.0)
    strength: float = Field(default=0.3, ge=0.0, le=1.0)
    format: Literal["square", "tall", "wide"] = "square"
    tshirt_mode: bool = False

    class Config:
        extra = Extra.forbid


# Create an argument parser from the Query model.
QUERY_PARSER = argparse.ArgumentParser()
QUERY_PARSER.add_argument("prompt", nargs="+")
assert sum(f.required for f in Query.__fields__.values()) == 1, "Just one required arg allowed"
for argname, field in Query.__fields__.items():
    if not field.required:
        if field.type_ == bool:
            QUERY_PARSER.add_argument(
                f"--{argname}", action=argparse.BooleanOptionalAction, default=False
            )
        else:
            QUERY_PARSER.add_argument(f"--{argname}")

# Create a usage string.
APP_USAGE_STR = QUERY_PARSER.format_help()
usage_str_lines = re.split(r"\n+", APP_USAGE_STR)
usage_str_lines = [line for line in usage_str_lines if "show this help message" not in line]
APP_USAGE_STR = "\n".join(usage_str_lines)
APP_USAGE_STR = APP_USAGE_STR.replace("[-h]", "")
APP_USAGE_STR = APP_USAGE_STR.replace("app.py", f"@{SLACK_APP_NAME}")


def parse_query(raw_query: str) -> Query:
    # We only consider the text after the @mention. Needs to be a non greedy
    # match so that we don't capture other slack formatting here.
    app_mention_match = re.search(r"(<@.*?>) (.*)", raw_query)
    if app_mention_match is None:
        raise ParseQueryException("You have to write a text prompt too!")

    # The second match group is the actual query for the app.
    query = app_mention_match.group(2)

    # We want to keep apostrophes around but they break the initial parsing
    # so they need to be escaped.
    query = query.replace("'", "\\'")

    # Parse query.
    try:
        args = QUERY_PARSER.parse_args(shlex.split(query))
        config = vars(args)
        config = {k: v for k, v in config.items() if v is not None}
    except:  # noqa: E722
        raise ParseQueryException("I couldn't parse that configuration!")

    # Special handling for some arguments.
    config["prompt"] = " ".join(config["prompt"])
    if "img_url" in config:
        config["img_url"] = remove_slack_link_formatting(config["img_url"])

    try:
        return Query(**config)
    except ValidationError:
        raise ParseQueryException("I couldn't validate those inputs!")


def remove_slack_link_formatting(x: str) -> str:
    m = re.search(r"<(.*?)>", x)
    if m is not None:
        return m.group(1)
    else:
        return x


def get_flags_string(config: Dict[str, Any]):
    config_strs = []
    for k, v in config.items():
        match v:
            case None:
                continue
            case bool():
                if v:
                    config_strs.append(f"--{k}")
            case str():
                config_strs.append(f'--{k}="{v}"')
            case _:
                config_strs.append(f"--{k}={v}")
    config_str = " ".join(config_strs)
    return config_str
