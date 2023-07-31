#!/bin/bash
set -eou pipefail

if [[ $(git symbolic-ref --short -q HEAD) != "main" ]]; then
    echo "You should be on the main branch"
    exit 1
fi

gcloud auth configure-docker
docker build -t gcr.io/embark-shared/ml2/john-stable-diffusion --target prod .
docker push gcr.io/embark-shared/ml2/john-stable-diffusion
