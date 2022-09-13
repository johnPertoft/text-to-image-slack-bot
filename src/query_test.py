import pytest
from pydantic import ValidationError

from .query import ParseQueryException
from .query import parse_query

# TODO:
# - Need to make the raw strings slack formatted
# - Handle additional spaces etc
# - Test for case insensitivity etc
# - Pytest can't access our dependencies
#   Maybe use pyenv or something instead?
# - Create some text fixture for creating the slack formatting
# - Add assertions on default values.


def test_just_prompt_str():
    raw = "<@burgerman> a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"


def test_simple_config():
    raw = "<@burgerman> seed=123, format=tall | a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.seed == 123
    assert q.format == "tall"


def test_config_with_uri():
    raw = "<@burgerman> img_uri=<https://test.url/img.png> | a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_uri == "https://test.url/img.png"


def test_config_with_uri_with_commas():
    raw = "<@burgerman> img_uri=<https://test.url/img.png?abc=1,2,3> | a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_uri == "https://test.url/img.png?abc=1,2,3"


def test_config_with_uri_last():
    raw = "<@burgerman> seed=123, img_uri=<https://test.url/img.png?abc=1,2,3> | a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_uri == "https://test.url/img.png?abc=1,2,3"
    assert q.seed == 123


def test_config_with_extra_spaces():
    raw = "<@burgerman> seed = 123 , img_uri= <https://test.url/img.png?abc=1,2,3> | a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_uri == "https://test.url/img.png?abc=1,2,3"
    assert q.seed == 123


def test_empty_config():
    raw = "<@burgerman> | a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"


def test_malformed_no_mention():
    raw = "a red apple"
    with pytest.raises(ParseQueryException):
        parse_query(raw)


def test_malformed_config():
    raw = "<@burgerman> ,,,,, | a red apple"
    with pytest.raises(ParseQueryException):
        parse_query(raw)


def test_undefined_config_item():
    raw = "<@burgerman> undefined_option=123 | a red apple"
    with pytest.raises(ValidationError):
        parse_query(raw)


def test_invalid_config_value():
    raw = "<@burgerman> seed=hmmm | a red apple"
    with pytest.raises(ValidationError):
        parse_query(raw)


def test_misspelled_img_uri():
    # Note: img_urL not img_urI here.
    raw = "<@burgerman> img_url=<https://test.url/img.png?abc=1,2,3> | a red apple"
    with pytest.raises(ValidationError):
        parse_query(raw)


def test_text_before_mention():
    raw = "here is some text followed by <@burgerman> seed=123, img_uri=<https://test.url/img.png?abc=1,2,3> | a red apple"  # noqa: E501
    q = parse_query(raw)
    assert q.prompt == "a red apple"
    assert q.img_uri == "https://test.url/img.png?abc=1,2,3"
    assert q.seed == 123
