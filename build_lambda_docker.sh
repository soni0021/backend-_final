#!/bin/bash

echo "Building Lambda package using Docker..."

# Build the Docker image
docker build -t lambda-builder .

# Create output directory
mkdir -p lambda_package

# Run container to copy the built package
docker run --rm -v $(pwd):/output lambda-builder cp -r /var/task /output/lambda_package

# Create the zip file
cd lambda_package && zip -r ../aws_lambda_artifact.zip .

echo "Lambda deployment package created: aws_lambda_artifact.zip"
echo "Upload this file to AWS Lambda and set the handler to 'main.handler'" 