#!/usr/bin/python

"""
This script simulates a post request as sent from Slack on app_mention events.
"""

import requests  # type: ignore

# TODO: Need to handle the slack api call back after generating somehow.
# TODO: Need to override the base url or something? Or just verify what it looks like
# or something.
# TODO: Check for slack api mock libs maybe.

# TODO: Take as input?
url = "https://b17d-34-90-75-154.eu.ngrok.io/slack/events"

# TODO: Need these in case we want to run with request verification.
# TODO: But can probably skip it.
# x-slack-signature
# x-slack-request-timestamp

headers = {"Content-Type": "application/json"}

# TODO: Set relevant values here.
body = {
    "token": "xxxxxx",
    "team_id": "TXXXXXXXX",
    "api_app_id": "A03U9439A5U",
    "event": {
        "type": "app_mention",
        "ts": "asdadadada",
        "event_ts": "1234567890.123456",
        "user": "UXXXXXXX1",
    },
    "type": "event_callback",
    "authorizations": [
        {"enterprise_id": "E12345", "team_id": "T12345", "user_id": "U12345", "is_bot": False}
    ],
    "event_context": "EC12345",
    "event_id": "Ev08MFMKH6",
    "event_time": 1234567890,
}

requests.post(url, headers=headers, json=body)
