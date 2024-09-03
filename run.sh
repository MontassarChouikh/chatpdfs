#!/bin/bash

# Path to your PDF file
file_path="1.pdf"

# URL of your Flask API running on the server
url="http://127.0.0.1:5001/analyze-pdf"

echo "Sending request to $url"

# Define the questions as a JSON string
questions='[{"field_name": "CertificateAuditor","question": "CertificateAuditor"}]'

response=$(curl -v "\nHTTP_STATUS_CODE: %{http_code}\n" \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@${file_path};type=application/pdf" \
  -F "questions=${questions}" \
  "$url")


# Print the response
echo "Response: $response"