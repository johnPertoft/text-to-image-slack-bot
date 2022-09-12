from .query import parse_query

# TODO:
# - Need to make the raw strings slack formatted
# - Handle additional spaces etc
# - Test for case insensitivity etc
# - Pytest can't access our dependencies
#   Maybe use pyenv or something instead?


def test_just_prompt_str():
    raw = "@burgerman a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"


def test_prompt_and_config():
    raw = "@burgerman seed=123, format=tall | a red apple"
    q = parse_query(raw)
    assert q.prompt == "a red apple"


def test_malformed():
    pass


def test_undefined_config_item():
    pass
