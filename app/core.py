import boto3
import json
import logging
import re
import time
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Initialize boto3 clients for Textract, S3, and Bedrock
textract_client = boto3.client('textract', region_name='eu-west-1')
s3_client = boto3.client('s3', region_name='eu-west-1')
bedrock_client = boto3.client('bedrock-runtime', region_name='eu-west-3')

def extract_text_from_pdf(s3_bucket, filename):
    """
    Extract text from a multi-page document using synchronous Textract API.
    """
    try:
        # Upload the file to S3
        s3_client.upload_file(filename, s3_bucket, filename)
        print(f"File {filename} successfully uploaded to bucket {s3_bucket}")
    except ClientError as e:
        print(f"Failed to upload file: {e}")
        return ""

    try:
        # Synchronously detect text using Textract
        response = textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': s3_bucket,
                    'Name': filename
                }
            }
        )

        # Dictionary to hold the extracted text by page
        pages_text = {}

        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                page_number = block.get('Page', 1)  # Default to page 1 if 'Page' is not available
                text_line = block['Text']

                # Append the text to the corresponding page's text list
                if page_number not in pages_text:
                    pages_text[page_number] = []
                pages_text[page_number].append(text_line)

        # Concatenate the text for each page, returning all pages as a single string
        all_pages_text = "\n\n".join([f"Page {page_num}:\n" + "\n".join(lines)
                                      for page_num, lines in pages_text.items()])

        # Print the extracted text to console
        print(f"Extracted text:\n{all_pages_text}")
        return all_pages_text

    except ClientError as e:
        if e.response['Error']['Code'] == 'UnsupportedDocumentException':
            print(f"Unsupported document format: {e}")
            return "Unsupported document format."
        print(f"Failed to extract text using Textract: {e}")
        return ""




def clean_model_output(raw_output):
    """
    Cleans the raw output by removing unnecessary escape characters and newlines.
    """
    cleaned_output = raw_output.replace("\\n", "").replace("\\", "")

    # Remove trailing commas (if any) that can break JSON parsing
    cleaned_output = re.sub(r",\s*}", "}", cleaned_output)  # Remove trailing commas before closing brace
    cleaned_output = re.sub(r",\s*]", "]", cleaned_output)  # Remove trailing commas before closing square bracket

    return cleaned_output

import time

def query_mistral_via_bedrock(extracted_text, questions, max_retries=10, retry_delay=2):
    """
    Query Mistral model via Bedrock API and retry the request until a valid JSON
    is received or max_retries is reached.
    """
    model_id = 'mistral.mistral-7b-instruct-v0:2'
    accept = 'application/json'
    contentType = 'application/json'

    # Construct the dynamic questions
    question_instructions = ""
    for question in questions:
        field_name = question['field_name']
        question_text = question['question']
        question_instructions += f'"{field_name}": "{question_text}", '

    # Remove the trailing comma and space
    question_instructions = question_instructions.rstrip(", ")

    # Example JSON format for guiding the model
    json_example = '''
    {
        "CertificateName": "Organic Cotton Production Certificate",  // type: text
        "CertificateType": "Production Certificate",  // type: enum(Production Certificate, Transaction Certificate)
        "CertificateAuditor": "Control Union Certifications",  // type: text
        "CertificateIssueDate": "01-01-2023",  // type: date in DD-MM-YYYY format
        "CertificateValidityStartDate": "01-01-2023",  // type: date in DD-MM-YYYY format
        "CertificateValidityEndDate": "31-12-2024",  // type: date in DD-MM-YYYY format
        "shipments": [
            {
                "ShipmentNo": "S001",
                "ShipmentDate": "15-02-2023",
                "GrossShippingWeight": 1200,
                "InvoiceReferences": "INV-101,
                "ShipmentDocNo": "INV-001"
            },
            {
                "ShipmentNo": "S002",
                "ShipmentDate": "25-03-2023",
                "GrossShippingWeight": 1500,
                "InvoiceReferences": "INV-104, INV-105, INV-106",
                "ShipmentDocNo": "INV-002"
            },
            {
                "ShipmentNo": "S003",
                "ShipmentDate": "10-05-2023",
                "GrossShippingWeight": 1800,
                "InvoiceReferences": "INV-107, INV-108, INV-109",
                "ShipmentDocNo": "INV-003" 
            }
        ]
    }
    '''

    prompt = f"""
    <s>[INST] {extracted_text}

    Please answer the following questions in valid JSON format. Ensure that:
    - All dates are formatted as DD-MM-YYYY.
    - Responses are structured in lists where appropriate.
    - Use NULL for missing or unavailable fields.
    - Make sure to include all shipments
    - Follow the example JSON format provided below.

    Example JSON format:
    {json_example}

    Now, answer the following questions:
    {{
      {question_instructions}
    }}
    Answer in JSON format. [/INST]
    """

    # Request body based on API specification
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50
    })

    attempt = 0
    while attempt < max_retries:
        try:
            # Send request to Bedrock to invoke Mistral
            response = bedrock_client.invoke_model(
                modelId=model_id,
                accept=accept,
                contentType=contentType,
                body=body
            )

            # Extract the text output from the Bedrock response
            response_body = json.loads(response['body'].read())
            result = response_body['outputs'][0]['text']

            # Clean the model's raw output to handle escape characters and newlines
            cleaned_result = clean_model_output(result)

            # Try to parse the cleaned result as JSON
            try:
                result_json = json.loads(cleaned_result)

                # Ensure all fields are checked for NULL values
                for question in questions:
                    field_name = question['field_name']
                    if field_name not in result_json or not result_json[field_name]:
                        result_json[field_name] = None  # Assign NULL if missing or empty

                # If valid JSON is obtained, return it
                return result_json

            except json.JSONDecodeError as e:
                logger.error(f"Model response is not valid JSON (attempt {attempt + 1}): {e}")
                
        except ClientError as e:
            logger.error(f"Failed to query Mistral model via Bedrock: {e}")
        
        # Increment the attempt counter and wait before retrying
        attempt += 1
        if attempt < max_retries:
            logger.info(f"Retrying... (attempt {attempt + 1})")
            time.sleep(retry_delay)
    
    # If retries fail, return None or a meaningful error
    logger.error("Max retries reached. Failed to get a valid JSON response.")
    return None
