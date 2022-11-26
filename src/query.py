import re
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Extra
from pydantic import Field
from pydantic import HttpUrl


class Query(BaseModel):
    prompt: str
    seed: Optional[int]
    img_url: Optional[HttpUrl]
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=15.0)
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    format: Literal["square", "tall", "wide"] = "square"
    nsfw_allowed: bool = False
    tshirt_mode: bool = False

    class Config:
        extra = Extra.forbid


class ParseQueryException(Exception):
    pass


def parse_query(raw_query: str) -> Query:
    # We only consider the text after the @mention. Needs to be a non greedy
    # match so that we don't capture other slack formatting here.
    app_mention_match = re.search(r"(<@.*?>) (.*)", raw_query)
    if app_mention_match is None:
        raise ParseQueryException("You have to write a text prompt too!")

    # The second match group is the actual query for the app.
    query = app_mention_match.group(2)

    # The query can optionally contain a configuration to circumvent
    # some or all of the default values.
    query_parts = query.split("|")
    if len(query_parts) == 1:
        prompt = query_parts[0].strip()
        return Query(prompt=prompt)
    else:
        # If "|" present in prompt part, join it back.
        prompt = "|".join(query_parts[1:])
        prompt = prompt.strip()

        config_str = query_parts[0]
        try:
            config = parse_config(config_str)
        except:  # noqa: E722
            raise ParseQueryException("I couldn't parse that configuration!")

        return Query(prompt=prompt, **config)


def parse_config(config_str: str) -> Dict[str, Any]:
    # We don't allow spaces to carry any significance here.
    config_str = config_str.replace(" ", "")

    # Early exit if no config supplied.
    if config_str == "":
        return dict()

    # Slack formats urls with angled brackets like <https://...>
    # Sometimes urls contain commas so need to take this into account
    # when we split by comma to separate each config item so we first
    # find these urls (should only be one).
    url_matches = re.finditer(r"<.*?>", config_str)
    url_spans = [url_match.span() for url_match in url_matches]

    # The valid commas to split at are the ones not within the url match
    # spans.
    all_comma_indices = [m.span()[0] for m in re.finditer(",", config_str)]
    split_indices = [i for i in all_comma_indices if not any(s1 <= i < s2 for s1, s2 in url_spans)]
    prev_idx = 0
    split_spans = []
    for split_idx in split_indices:
        split_spans.append((prev_idx, split_idx))
        prev_idx = split_idx + 1
    split_spans.append((prev_idx, None))  # type: ignore

    # Split into each config item and create a config dict.
    config_records = [config_str[i:j] for i, j in split_spans]
    config = dict([c.split("=", 1) for c in config_records])

    def maybe_remove_slack_formatting(x: str) -> str:
        m = re.search(r"<(.*?)>", x)
        if m is not None:
            return m.group(1)
        else:
            return x

    # Remove any Slack formatting brackets
    config = {k: maybe_remove_slack_formatting(v) for k, v in config.items()}

    return config
