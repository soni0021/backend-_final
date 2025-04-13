#!/bin/bash

# Create a temporary directory for dependencies
mkdir -p lambda_package/dependencies

# Install dependencies in the temp directory
pip3 install -t lambda_package/dependencies -r requirements.txt

# Create the ZIP file with dependencies
cd lambda_package/dependencies
zip -r ../../aws_lambda_artifact.zip .
cd ../..

# Add application files to the ZIP
zip -u aws_lambda_artifact.zip main.py
zip -u aws_lambda_artifact.zip Corrected_Marks_vs_Rank.xlsx

# Add the cleaned_data directory and its contents
cd cleaned_data
zip -r ../aws_lambda_artifact.zip .
cd ..

echo "Lambda deployment package created: aws_lambda_artifact.zip" 