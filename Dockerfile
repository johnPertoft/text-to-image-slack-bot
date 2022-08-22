# TODO: Add a separate dockerfile for deployment to cloud run or something?
# TODO: Maybe use multi stage build between this and the devcontainer dockerfile.
# TODO: Build via cloud build?
# TODO: Can't have gpus with cloud run I think.
# TODO: Probably need to setup gke cluster with gpu nodes instead?
# TODO: Would also be faster to not have to load model every time?
# TODO: Also restrict ip ranges to only allow from gcloud?

FROM python:3.10
RUN pip install \
    google-cloud-secret-manager \
    slack_bolt
WORKDIR /workspace
COPY src src
COPY img.png img.png
ENTRYPOINT ["python", "-m", "src.app"]
