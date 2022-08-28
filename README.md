# text-to-image-slack-bot
Put secrets in `.envrc`, they will be automatically loaded by `direnv`.

## TODO
- [ ] Figure out why regular python/ubuntu image is not working on ml cluster with gpus.
- [ ] Have a single Dockerfile, with multiple build targets for dev container and prod.
- [ ] Automatic build/deploys via CI.
- [ ] Add option to pass in arguments to model inference.
- [ ] Restrict ip ranges for requests?
- [ ] Use slack SocketMode instead?
- [ ] Fix temporary names like john-text- prefixes
- [ ] Save all generated images in some bucket/bigquery?
- [ ] Maybe build via cloud build?
- [ ] Add support for image as input?
- [ ] Add support for image inpainting?

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

## Derp derp
```bash
# TODO: missing credentials for this depending on service account?
gcloud builds submit --tag gcr.io/embark-nlp/john-test
```
