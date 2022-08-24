# text-to-image-slack-bot
Put secrets in `.envrc`, they will be automatically loaded by `direnv`.

## TODO
- Put model files in container / gcs
- Use AsyncApp instead?
- Use SocketMode?
- Maybe some kind of setup with sending jobs to a queue and give some feedback immediately?
- Sometimes getting "ghost" responses, some missing ack or something?

## Setup
<details>
<summary>Click to expand</summary>

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

## How to run

1. ```bash
   python -m src.app
   ```
2. ```bash
   # ngrok config add-authtoken <token>

   ngrok http 3000
   ```
3. Update request url for event subscription: https://api.slack.com/apps/A03U9439A5U/event-subscriptions?
