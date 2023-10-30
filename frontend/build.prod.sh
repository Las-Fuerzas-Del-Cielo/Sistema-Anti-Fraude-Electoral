#!/bin/bash
##author: Samuel Onti <enrissmuelo@gmail.com>
DOCKER_IMAGE="app-frontend-prod"
CONTAINER_NAME="temp-container"

docker build -t $DOCKER_IMAGE -f Dockerfile.prod .

docker run --name $CONTAINER_NAME -d $DOCKER_IMAGE tail -f /dev/null

docker cp $CONTAINER_NAME:/app/dist/. ./dist

docker rm $CONTAINER_NAME