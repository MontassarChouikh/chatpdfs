#!/bin/bash

# This script sends a PDF file and a list of questions to the Flask API

API_URL="http://127.0.0.1:5200/analyze-pdf"
PDF_FILE="1.pdf"  # Replace with the actual path to your test PDF file

# Questions in JSON format
QUESTIONS='[
        {"field_name": "CertificateName", "question": "name of the certificate"},
        {"field_name": "CertificateType", "question": "type of the certificate"},
        {"field_name": "CertificateAuditor", "question": "auditor, certificator or certification body who issued the certificate"},
        {"field_name": "CertificateIssueDate", "question": "date of issuance of the certificate"},
        {"field_name": "CertificateValidityStartDate", "question": "validity start date of the certificate"},
        {"field_name": "CertificateValidityEndDate", "question": "validity end date of the certificate"},
        {"field_name": "certified products", "question": "Provide all certified products details"},
        {"field_name": "shipments", "question": "Provide all shipments details"}    
      ]'

# Make the POST request using curl
curl -X POST "$API_URL" \
  -F "file=@$PDF_FILE" \
  -F "questions=$QUESTIONS"
