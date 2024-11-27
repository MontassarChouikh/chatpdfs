#!/bin/bash

# Path to your PDF file
file_path="TC.pdf"

# URL of your Flask API running on the server
url="http://127.0.0.1:5001/process-pdf"

echo "Sending request to $url"

# Define the questions as a JSON string
questions='[{"field_name": "CertificateAuditor","question": "CertificateAuditor"},
            {"field_name": "CertificateValidityStartDate","question": "CertificateValidityStartDate"},
            {"field_name": "CertificateRawMaterials","question": "CertificateRawMaterials"},
            {"field_name": "shipments", "question": "shipments"},
            {"field_name": "products", "question": "products"}
            ]'

# Send the POST request using curl
response=$(curl -v -o response.txt -w "\nHTTP_STATUS_CODE: %{http_code}\n" \
  -F "file=@${file_path};type=application/pdf" \
  -F "questions=${questions}" \
  "$url" 2>&1)

# Print the entire response for debugging
echo "Response:"
echo "$response"

# Extract the HTTP status code from the response
http_status=$(echo "$response" | grep "HTTP_STATUS_CODE:" | awk '{print $2}')

# Print the HTTP status code and response body
echo "HTTP Status: $http_status"
cat response.txt 2>/dev/null

# Clean up
rm response.txt 2>/dev/null
