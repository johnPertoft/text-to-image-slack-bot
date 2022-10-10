"""
This script simulates a post request as sent from Slack on app_mention events.
"""

import json
import os
import shlex
import subprocess
import sys

import requests  # type: ignore

# TODO:
# - Pass proper headers, like x-slack-signature and x-slack-request-timestamp
# - Then run app with request verification on
# - Need to pass some mock slack client to verify what should happen.
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

if len(sys.argv) > 1:
    query = shlex.join(sys.argv[1:])
else:
    # TODO: Links can be formatted with a display text
    # like <https://google.com|Link text>
    query = "img_url=<https://google.com/img.png> | a horse in space"

event_text = f"<@bot-id> {query}"

# Note: This seems to be the minimal body required to reach the app_mention handler function.
# See https://api.slack.com/apis/connections/events-api#receiving_events
# for how this body is actually expected to look.
body = {
    "team_id": "dummy-team-id",
    "channel_id": "dummy-channel-id",
    "type": "event_callback",
    "event": {
        "type": "app_mention",
        "ts": "dummy-event-ts",
        "event_ts": "1234567890.123456",
        "user": "dummy-user-id",
        "text": event_text,
    },
}

resp = requests.post(url, headers=headers, json=body)
print(resp)
