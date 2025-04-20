#!/bin/bash

# Define the categories
categories=("Open" "OBC" "EWS" "SC" "ST" "Open-PwD" "OBC-PwD" "SC-PwD" "EWS-PwD" "ST-PwD")

# Define the base rank for each category
declare -A ranks
ranks["Open"]=5000
ranks["OBC"]=10000
ranks["EWS"]=7500
ranks["SC"]=15000
ranks["ST"]=20000
ranks["Open-PwD"]=25000
ranks["OBC-PwD"]=30000
ranks["SC-PwD"]=40000
ranks["EWS-PwD"]=35000
ranks["ST-PwD"]=45000

# Define the counseling round
round="1"

# Loop through each category and find colleges
for category in "${categories[@]}"; do
    echo "======================================================================"
    echo "Finding colleges for Category: $category with rank: ${ranks[$category]}"
    echo "======================================================================"
    
    # Make the API call
    curl -s -X POST -H "Content-Type: application/json" \
         -d "{\"state\": \"All India\", \"category\": \"$category\", \"round\": \"$round\", \"rank\": ${ranks[$category]}}" \
         http://localhost:8081/find_colleges | jq
    
    echo ""
    echo ""
done 