# Text to Image Slack Bot (Stable-diffusion)

![Lint status](https://github.com/johnPertoft/text-to-image-slack-bot/actions/workflows/lint.yaml/badge.svg?branch=main)
![Test status](https://github.com/johnPertoft/text-to-image-slack-bot//actions/workflows/test.yaml/badge.svg?branch=main)

## Basic information
|   |   |
|---|---|
|Slack application home|https://api.slack.com/apps/A03U9439A5U/|
|container image|`gcr.io/embark-shared/ml2/john-stable-diffusion`|

## Setup
<details>
<summary>Click to expand</summary>

### Local environment secrets
Put secrets in `.envrc`, they will be automatically loaded by `direnv`.

### Create secrets
```bash
gcloud secrets create john-test-slack-bot-token
gcloud secrets create john-test-slack-signing-secret
```

### Update secret versions
```bash
echo -n $SLACK_BOT_TOKEN | gcloud secrets versions add john-test-slack-bot-token --data-file=-
echo -n $SLACK_SIGNING_SECRET | gcloud secrets versions add john-test-slack-signing-secret --data-file=-
```

### Allow default service account to access secrets
```bash
gcloud secrets add-iam-policy-binding john-test-slack-bot-token \
   --role roles/secretmanager.secretAccessor \
   --member serviceAccount:153639231195-compute@developer.gserviceaccount.com

gcloud secrets add-iam-policy-binding john-test-slack-signing-secret \
   --role roles/secretmanager.secretAccessor \
   --member serviceAccount:153639231195-compute@developer.gserviceaccount.com
```

</details>

## How to run locally
<details>
<summary>Click to expand</summary>

1. ```bash
   python -m src.app
   ```
2. ```bash
   # ngrok config add-authtoken <token>

   ngrok http 3000
   ```
3. Update request url for event subscription: https://api.slack.com/apps/A03U9439A5U/event-subscriptions?

</details>

## Deploying
Currently the app is deployed by
- Building and pushing the new image to gcr, just run `./deploy.sh` in the project root.
- Manually deleting the pod to have it automatically restart and use the new image
