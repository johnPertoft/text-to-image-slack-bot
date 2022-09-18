#!/bin/bash
set -eu

# Make sure that the docker group inside the container has the same
# id as the docker group on the host. This depends on the docker socket
# being mounted.
HOST_DOCKER_GID=$(stat -c "%g" /var/run/docker.sock)
sudo groupmod -g $HOST_DOCKER_GID docker
newgrp docker
