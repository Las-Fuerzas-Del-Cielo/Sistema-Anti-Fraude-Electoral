#!/bin/bash

# Define the name of the Docker image and the name of the container
DOCKER_IMAGE="app-frontend"
CONTAINER_NAME="front-lla"
CONTAINER_ID=$(docker ps -q -a -f name=$CONTAINER_NAME)

# Check if the container exists and then remove it if it does
if [ -z "$CONTAINER_ID" ]; then

  # Container already exists, take appropriate action or skip
  echo "Container $CONTAINER_NAME doesn't exist. Skipping docker stop and docker rm."
else

  # Stop and remove the container
  docker stop $CONTAINER_ID
  docker rm $CONTAINER_ID

  # Remove the image
  docker rmi $DOCKER_IMAGE
fi

# Build the Docker image
docker build -t $DOCKER_IMAGE -f Dockerfile.dev .

# Run the container with the application
docker run -d --name $CONTAINER_NAME -p 8080:5173 $DOCKER_IMAGE

echo -e "\nAccess the application in development mode at http://localhost:8080"