#!/bin/bash
set -e

if [ "$CI" = "true" ]; then
    echo "Skipping post-create command because this is running on CI."
    exit
fi

pre-commit install

touch .envrc
direnv allow

# Make sure that the docker group inside the container has the same
# id as the docker group on the host. This depends on the docker socket
# being mounted.
HOST_DOCKER_GID=$(stat -c "%g" /var/run/docker.sock)
sudo groupmod -g $HOST_DOCKER_GID docker
newgrp docker

# Note: This seems to work but no commands after the newgrp command above
# gets run. I'm assuming that has something to do with how newgrp drops you
# into a new shell?
# This also seems to break post attach commands unfortunately.
