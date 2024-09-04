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

# Define your S3 bucket name
S3_BUCKET = 'your-s3-bucket-name'

def convert_pdf_to_images(pdf_file, output_folder='output_images'):
    """
    Convert each page of a PDF (in-memory) into an image using pdf2image.
    """
    try:
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
    Extract text and confidence scores from an image using AWS Textract.
    """
    try:
        response = textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': s3_bucket,
                    'Name': image_file
                }
            }
        )

        extracted_text = ""
        confidence_scores = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                extracted_text += block['Text'] + "\n"
                confidence_scores.append(block['Confidence'])

        logger.info(f"Extracted text from {image_file}:\n{extracted_text}")
        logger.info(f"Confidence scores from {image_file}: {confidence_scores}")
        return extracted_text, confidence_scores
    except ClientError as e:
        logger.error(f"Failed to extract text from {image_file}: {e}")
        return "", []

def clean_text(text):
    """
    Cleans the extracted text by removing unnecessary escape characters and newlines.
    """
    return text.replace("\\n", "").replace("\\", "")

def extract_text_from_pdf(s3_bucket, pdf_file_stream, output_folder='output_images'):
    """
    Extract text and confidence scores from a multi-page PDF by converting it to images, 
    uploading to S3, and using AWS Textract to extract text from each image.
    """
    try:
        image_files = convert_pdf_to_images(pdf_file_stream, output_folder)
        
        if not image_files:
            logger.error("No images generated from the PDF.")
            return "", 0.0
        
        all_text = []
        all_confidence_scores = []
        
        for image_file in image_files:
            uploaded = upload_image_to_s3(s3_bucket, image_file)
            if uploaded:
                extracted_text, confidence_scores = extract_text_from_image(s3_bucket, image_file)
                if extracted_text:
                    cleaned_text = clean_text(extracted_text)
                    all_text.append(cleaned_text)
                    all_confidence_scores.extend(confidence_scores)

        # Combine text from all images
        combined_text = "\n\n".join(all_text)
        logger.info(f"Combined extracted text:\n{combined_text}")

        # Calculate the average confidence score
        if all_confidence_scores:
            average_confidence_score = sum(all_confidence_scores) / len(all_confidence_scores)
        else:
            average_confidence_score = 0.0

        logger.info(f"Average confidence score: {average_confidence_score}")

        # Clean up the local files after processing
        for image_file in image_files:
            os.remove(image_file)

        return combined_text, average_confidence_score
    except Exception as e:
        logger.error(f"An error occurred during the PDF text extraction process: {e}")
        return "", 0.0

def clean_model_output(raw_output):
    """
    Cleans the raw output by removing unnecessary escape characters and newlines.
    """
    cleaned_output = raw_output.replace("\\n", "").replace("\\", "")
    cleaned_output = re.sub(r",\s*}", "}", cleaned_output)
    cleaned_output = re.sub(r",\s*]", "]", cleaned_output)
    return cleaned_output

def query_claude_via_bedrock(extracted_text, questions, image_path=None, max_retries=10, retry_delay=2):
    """
    Query the Claude model via Bedrock with extracted text and dynamic questions.
    """
    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    accept = 'application/json'
    contentType = 'application/json'

    question_instructions = ", ".join([f'"{q["field_name"]}": "{q["question"]}"' for q in questions])

    prompt = f"""
    You are given the extracted text from a document. Please answer the following questions based on the provided text.
    
    The document may contain details such as:
    - Certificate details (name, type, auditor, issue date, validity dates)
    - Shipments (shipment number, date, gross shipping weight, invoice references, etc.)

    Use the format below to answer:
    - All dates must be in the format YYYY-MM-DD.
    - If any information is missing or unavailable, use NULL.
    - Ensure that all data is provided in the requested JSON format.

    Extracted text:
    {extracted_text}

    Now, answer the following questions:
    {{
      {question_instructions}
    }}

    Provide the answer in JSON structure:
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

    attempt = 0
    while attempt < max_retries:
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                accept=accept,
                contentType=contentType,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            logger.info(f"Full response from Claude: {response_body}")

            result = response_body['content'][0]['text']
            cleaned_result = clean_model_output(result)
            return json.loads(cleaned_result)

        except json.JSONDecodeError as e:
            logger.error(f"Model response is not valid JSON (attempt {attempt + 1}): {e}")
        except ClientError as e:
            logger.error(f"Failed to query Claude model via Bedrock: {e}")

        attempt += 1
        if attempt < max_retries:
            logger.info(f"Retrying... (attempt {attempt + 1})")
            time.sleep(retry_delay)

    logger.error("Max retries reached. Failed to get a valid JSON response.")
    return None
