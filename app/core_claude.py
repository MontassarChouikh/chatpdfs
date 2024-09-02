import os
import boto3
import json
import logging
import re
import time
from pdf2image import convert_from_bytes
from botocore.exceptions import ClientError
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Initialize boto3 clients for Textract, S3, and Bedrock
textract_client = boto3.client('textract', region_name='eu-west-1')
s3_client = boto3.client('s3', region_name='eu-west-1')
bedrock_client = boto3.client('bedrock-runtime', region_name='eu-west-3')

def convert_pdf_to_images(pdf_file, output_folder='output_images'):
    """
    Convert each page of a PDF (in-memory) into an image using pdf2image.
    """
    try:
        # Convert PDF to images using the in-memory file
        images = convert_from_bytes(pdf_file)
        
        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_filenames = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"page_{i+1}.png")
            image.save(image_path, 'PNG')
            logger.info(f"Page {i+1} saved as image: {image_path}")
            image_filenames.append(image_path)

        return image_filenames
    except Exception as e:
        logger.error(f"An error occurred during PDF to image conversion: {e}")
        return []

def upload_image_to_s3(s3_bucket, image_file):
    """
    Upload an image to S3.
    """
    try:
        s3_client.upload_file(image_file, s3_bucket, image_file)
        logger.info(f"File {image_file} successfully uploaded to bucket {s3_bucket}")
        return True
    except ClientError as e:
        logger.error(f"Failed to upload file {image_file} to S3: {e}")
        return False

def extract_text_from_image(s3_bucket, image_file):
    """
    Extract text from an image using AWS Textract.
    """
    try:
        # Detect text from the image using Textract
        response = textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': s3_bucket,
                    'Name': image_file
                }
            }
        )

        # Extract text from the response
        extracted_text = ""
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                extracted_text += block['Text'] + "\n"

        logger.info(f"Extracted text from {image_file}:\n{extracted_text}")
        return extracted_text
    except ClientError as e:
        logger.error(f"Failed to extract text from {image_file}: {e}")
        return ""

def clean_text(text):
    """
    Cleans the extracted text by removing unnecessary escape characters and newlines.
    """
    return text.replace("\\n", "").replace("\\", "")

def extract_text_from_pdf(s3_bucket, pdf_file_stream, output_folder='output_images'):
    """
    Extract text from a multi-page PDF by converting it to images, uploading to S3,
    and using AWS Textract to extract text from each image.
    """
    try:
        # Convert PDF to images (from in-memory stream)
        image_files = convert_pdf_to_images(pdf_file_stream, output_folder)
        
        if not image_files:
            logger.error("No images generated from the PDF.")
            return ""
        
        all_text = []
        for image_file in image_files:
            # Upload image to S3
            uploaded = upload_image_to_s3(s3_bucket, image_file)
            if uploaded:
                # Extract text from the image using Textract
                extracted_text = extract_text_from_image(s3_bucket, image_file)
                if extracted_text:
                    # Clean the extracted text
                    cleaned_text = clean_text(extracted_text)
                    all_text.append(cleaned_text)
        
        # Combine text from all images
        combined_text = "\n\n".join(all_text)
        logger.info(f"Combined extracted text:\n{combined_text}")
        return combined_text
    except Exception as e:
        logger.error(f"An error occurred during the PDF text extraction process: {e}")
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

def query_claude_via_bedrock(extracted_text, questions, image_path=None, max_retries=10, retry_delay=2):
    """
    Query the Claude model via Bedrock with extracted text and dynamic questions.
    """
    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    accept = 'application/json'
    contentType = 'application/json'

    # Construct the dynamic questions (received from the user in curl)
    question_instructions = ", ".join([f'"{q["field_name"]}": "{q["question"]}"' for q in questions])

    # Create the prompt for Claude
    prompt = f"""
    You are given the extracted text from a document. Please answer the following questions based on the provided text.
    
    The document may contain details such as:
    - Certificate details (name, type, auditor, issue date, validity dates)
    - Shipments (shipment number, date, gross shipping weight, invoice references, etc.)

    Use the format below to answer:
    - All dates must be in the format DD-MM-YYYY.
    - If any information is missing or unavailable, use NULL.
    - Ensure that all data is provided in the requested JSON format.

    Extracted text:
    {extracted_text}

    Now, answer the following questions:
    {{
      {question_instructions}
    }}

    Provide the answer in the following JSON structure:
    {{
        "CertificateName": "string",
        "CertificateType": "string",
        "CertificateAuditor": "string",
        "CertificateIssueDate": "DD-MM-YYYY",
        "CertificateValidityStartDate": "DD-MM-YYYY",
        "CertificateValidityEndDate": "DD-MM-YYYY",
        "shipments": [
            {{
            }}
        ]
    }}
    """

    # Prepare the image data if provided
    image_data = None
    if image_path:
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    # Construct the request body for Claude
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    # If image data is provided, add it to the request
    if image_data:
        body["messages"][0]["content"].insert(0, {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data
            }
        })

    # Retry loop for querying the Claude model
    attempt = 0
    while attempt < max_retries:
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                accept=accept,
                contentType=contentType,
                body=json.dumps(body)
            )

            # Extract and log the response from Claude
            response_body = json.loads(response['body'].read())
            logger.info(f"Full response from Claude: {response_body}")

            result = response_body['content'][0]['text']

            # Clean and parse the output
            cleaned_result = clean_model_output(result)
            return json.loads(cleaned_result)

        except json.JSONDecodeError as e:
            logger.error(f"Model response is not valid JSON (attempt {attempt + 1}): {e}")
        except ClientError as e:
            logger.error(f"Failed to query Claude model via Bedrock: {e}")

        # Retry logic
        attempt += 1
        if attempt < max_retries:
            logger.info(f"Retrying... (attempt {attempt + 1})")
            time.sleep(retry_delay)

    logger.error("Max retries reached. Failed to get a valid JSON response.")
    return None
