#!/bin/bash

# TODO: temporary script for deploying image.
# - Should live on CI instead with proper tagging etc.
# - Maybe use cloud build instead?
# - Automatically download the pipeline files?

if [ -d "pipelines/sd-pipeline" ]; then
    docker build -t gcr.io/embark-shared/ml2/john-stable-diffusion .
    docker push gcr.io/embark-shared/ml2/john-stable-diffusion
else
    echo "Download the model files first"
    exit 1
fi
