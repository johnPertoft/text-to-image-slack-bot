#!/bin/bash
set -e
set -u
set -o pipefail

# TODO: temporary script for deploying image.
# - Should live on CI instead with proper tagging etc.
# - Maybe use cloud build instead?
# - Automatically download the pipeline files?

if [[ $(git symbolic-ref --short -q HEAD) != "main" ]]; then
    echo "You should be on the main branch"
    exit 1
fi

exit

if [ -d "pipelines/sd-pipeline" ]; then
    gcloud auth configure-docker
    docker build -t gcr.io/embark-shared/ml2/john-stable-diffusion .
    docker push gcr.io/embark-shared/ml2/john-stable-diffusion
else
    echo "Download the model files first"
    exit 1
fi
