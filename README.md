# text-to-image-slack-bot
Put secrets in `.envrc`, they will be automatically loaded by `direnv`.

1. ```bash
   python -m src.app
   ```
2. ```bash
   # ngrok config add-authtoken <token>

   ngrok http 3000
   ```
3. Update request url for event subscription: https://api.slack.com/apps/A03U9439A5U/event-subscriptions?
