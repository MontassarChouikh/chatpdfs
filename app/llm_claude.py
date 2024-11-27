import os
import json
import logging
import time
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeBedrockAPI:
    def __init__(self):
        self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        self.region_name = os.getenv('AWS_REGION', 'eu-west-3')  # Correct region
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region_name)

    def validate_json(self, response_text):
        """
        Validate and parse JSON response, removing markdown if necessary.
        """
        try:
            # Strip markdown tags if present
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[7:-3].strip()

            # Attempt to parse the cleaned JSON
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None

    def query_claude(self, extracted_text, questions, prefilled_response=None, max_tokens=8000, max_retries=3, retry_delay=2):
        question_instructions = ", ".join([f'"{q["field_name"]}": "{q["question"]}"' for q in questions])

        system_prompt = "You are given the extracted text from a document. Please answer the following questions based on the provided text."

        json_prefill = """
        {
            "CertificateName": "",
            "CertificateType": "",
            "CertificateAuditor": "",
            "CertificateIssueDate": "",
            "CertificateValidityStartDate": "",
            "CertificateValidityEndDate": "",
            "shipments": []
        }
        """

        user_message_content = f"""
        The document may contain details such as:
        - Certificate details (name, type, auditor, issue date, validity dates)
        - Shipments (shipment number, date, gross shipping weight, invoice references, etc.)

        Use the format below to answer:
        - All dates must be in the format YYYY-MM-DD.
        - If any information is missing or unavailable, use NULL.
        - Answer with only words, not full sentences.
        - Ensure that all data is provided in the requested JSON format.

        Extracted text:
        {extracted_text}

        Now, answer the following questions:
        {{
        {question_instructions}
        }}

        Provide the answer in JSON structure like in this example:
        ```json
        {json_prefill}
        """

        messages = [{"role": "user", "content": user_message_content}]

        if prefilled_response:
            assistant_message = {"role": "assistant", "content": prefilled_response}
            messages.append(assistant_message)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": messages,
            "max_tokens": max_tokens
        })

        attempt = 0
        while attempt < max_retries:
            try:
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    contentType='application/json',
                    accept='application/json'
                )

                response_body = json.loads(response.get('body').read())
                logger.info(f"Full response from Claude: {response_body}")

                result = response_body.get('content', [{}])[0].get('text', '')

                # Validate and parse the JSON response
                validated_json = self.validate_json(result)
                if validated_json:
                    return validated_json
                else:
                    logger.warning(f"Attempt {attempt + 1}: Invalid JSON output. Retrying...")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error querying Claude: {e}")
            
            attempt += 1
            if attempt < max_retries:
                time.sleep(retry_delay)

        logger.error("Max retries reached. Failed to get a valid response.")
        return None
