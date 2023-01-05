import pytest

from .query import ParseQueryException
from .query import parse_query


def test_just_prompt_str():
    raw = "<@burgerman> a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"


def test_simple_config():
    raw = "<@burgerman> a red apple --format=tall --seed=123"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.seed == 123
    assert q.format == "tall"


def test_simple_config_prompt_last():
    raw = "<@burgerman> --format=tall --seed=123 a red apple "
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.seed == 123
    assert q.format == "tall"


def test_config_with_uri():
    raw = "<@burgerman> a red apple --img_url=<https://test.url/img.png>"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_url == "https://test.url/img.png"


def test_config_with_uri_with_commas():
    raw = "<@burgerman> a red apple --img_url=<https://test.url/img.png?abc=1,2,3>"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_url == "https://test.url/img.png?abc=1,2,3"


def test_config_with_uri_last():
    raw = "<@burgerman> a red apple --seed=123 --img_url=<https://test.url/img.png?abc=1,2,3>"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_url == "https://test.url/img.png?abc=1,2,3"
    assert q.seed == 123


def test_config_with_negative_prompt():
    raw = '<@burgerman> a red apple --negative_prompt="green banana, blue"'
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.negative_prompt == "green banana, blue"


def test_config_with_apostrophe():
    raw = "<@burgerman> an apple's banana"
    q = parse_query(raw)
    assert q.prompt == "an apple's banana"


def test_malformed_no_mention():
    raw = "a red apple"
    with pytest.raises(ParseQueryException):
        parse_query(raw)


def test_undefined_config_item():
    raw = "<@burgerman> a red apple --undefined_option=123"
    with pytest.raises(ParseQueryException):
        parse_query(raw)


def test_invalid_config_value():
    raw = "<@burgerman> a red apple --seed=hmmm"
    with pytest.raises(ParseQueryException):
        parse_query(raw)


def test_misspelled_img_url():
    # Note: img_uri not img_urL here.
    raw = "<@burgerman> a red apple --img_uri=<https://test.url/img.png?abc=1,2,3>"
    with pytest.raises(ParseQueryException):
        parse_query(raw)


def test_text_before_mention():
    raw = "here is some text followed by <@burgerman> a red apple --seed=123 --img_url=<https://test.url/img.png?abc=1,2,3>"  # noqa: E501
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_url == "https://test.url/img.png?abc=1,2,3"
    assert q.seed == 123
