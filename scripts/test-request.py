#!/usr/bin/python

"""
This script simulates a post request as sent from Slack on app_mention events.
"""

import json
import os
import subprocess

import requests  # type: ignore

# TODO:
# - Pass proper headers, like x-slack-signature and x-slack-request-timestamp
# - Then run app with request verification on
# - Need to pass some mock slack client.
# - How to generate a realistic slack formatted input text? E.g. link formatting etc.
#   See https://api.slack.com/reference/surfaces/formatting#escaping


assert "NGROK_API_KEY" in os.environ, "NGROK_API_KEY environment variable expected"
ngrok_api_key = os.environ["NGROK_API_KEY"]
ngrok_output_str = subprocess.check_output(
    f"ngrok api tunnels list --api-key {ngrok_api_key}", stderr=subprocess.DEVNULL, shell=True
)
ngrok_output = json.loads(ngrok_output_str)
assert len(ngrok_output["tunnels"]) > 0, "Is ngrok running?"
public_url = ngrok_output["tunnels"][0]["public_url"]
url = f"{public_url}/slack/events"

headers = {"Content-Type": "application/json"}

# Note: This seems to be the minimal body required to reach the app_mention handler function.
# See https://api.slack.com/apis/connections/events-api#receiving_events
# for how this body is actually expected to look.
body = {
    "team_id": "dummy-team-id",
    "type": "event_callback",
    "event": {
        "type": "app_mention",
        "ts": "dummy-event-ts",
        "event_ts": "1234567890.123456",
        "user": "dummy-user-id",
    },
}

resp = requests.post(url, headers=headers, json=body)

# TODO: Even if the handler code fails, this is a 200.
# Maybe from some ack() in a middleware function?
# TODO: How do we verify that everything "works"?
print(resp)
