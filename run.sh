#!/bin/bash

# Path to your PDF file
file_path="chanel2.pdf"

# URL of your Flask API running on the server
url="https://chatpdf.crystalchain.io/process-pdf"

echo "Sending request to $url"

# Define the questions as a JSON string
questions='[{"field_name": "CertificateAuditor","question": "CertificateAuditor"},
            {"field_name": "shipments", "question": "shipments"},
            {"field_name": "products", "question": "products"}
            ]'

# Send the POST request using curl
response=$(curl -s -o response.txt -w "\nHTTP_STATUS_CODE: %{http_code}\n" \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@${file_path};type=application/pdf" \
  -F "questions=${questions}" \
  "$url")

# Extract the HTTP status code from the response
http_status=$(echo "$response" | grep "HTTP_STATUS_CODE:" | awk '{print $2}')

# Print the HTTP status code and response body
echo "HTTP Status: $http_status"
cat response.txt

# Clean up
rm response.txt
