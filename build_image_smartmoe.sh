#!/bin/bash

# Get current user's ID and group ID
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(whoami)

# Build the Docker image
docker build \
  --build-arg USER_ID=$USER_ID \
  --build-arg GROUP_ID=$GROUP_ID \
  --build-arg USER_NAME=$USER_NAME \
  -f Dockerfile-smartmoe \
  -t moe_smartmoe:latest .