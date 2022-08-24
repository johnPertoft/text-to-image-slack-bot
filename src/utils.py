from google.cloud import secretmanager


def get_secret(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    full_secret_name = f"projects/embark-nlp/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=full_secret_name)
    return response.payload.data.decode("UTF-8")
