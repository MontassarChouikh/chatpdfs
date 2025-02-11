import os
import json
import time
import logging
import boto3
import re
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralBedrockAPI:
    def __init__(self):
        self.model_id = 'mistral.mistral-7b-instruct-v0:2'
        self.region_name = os.getenv('AWS_REGION', 'eu-west-3')
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region_name)

        # Cost per token (Mistral pricing)
        self.cost_per_input_token = 0.00055 / 1000  # $0.00055 per 1K input tokens
        self.cost_per_output_token = 0.00165 / 1000  # $0.00165 per 1K output tokens

    def query_mistral(self, extracted_text, questions, max_retries=3, retry_delay=2):
        """Query Mistral model with extracted text and questions while logging token cost."""

        # Format questions
        question_instructions = ", ".join([f'"{q["field_name"]}": "{q["question"]}"' for q in questions])

        # Construct prompt
        prompt = f"""
        You are given the extracted text from a document.
        Use the format below to answer:
        - All dates must be in the format YYYY-MM-DD.
        - If any information is missing or unavailable, use NULL.  
        Please answer the following questions based on the provided text.
        Extracted text:
        {extracted_text}
        Now, answer the following questions:
        {{
          {question_instructions}
        }}
        Provide the answer in JSON structure like in this example:
        {{
            "CertificateName": "string",
            "CertificateType": "string",
            "CertificateAuditor": "string",
            "CertificateIssueDate": "YYYY-MM-DD",
            "CertificateValidityStartDate": "YYYY-MM-DD",
            "CertificateValidityEndDate": "YYYY-MM-DD",
            "shipments": []
        }}
        """

        # Estimate input token count (1 token ≈ 4 characters)
        estimated_input_tokens = len(prompt) // 4

        # Payload
        body = {
            "prompt": prompt,
            "max_tokens": 8000,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50
        }

        attempt = 0
        while attempt < max_retries:
            try:
                # Invoke model
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    accept="application/json",
                    contentType="application/json",
                    body=json.dumps(body)
                )

                # Parse response
                response_body = response['body'].read().decode('utf-8')
                logger.info(f"Raw response from Mistral: {response_body}")

                try:
                    # Extract and clean JSON response
                    response_data = json.loads(response_body)
                    outputs = response_data.get('outputs', [])

                    if outputs:
                        raw_text = outputs[0].get('text', '')

                        # Estimate output tokens
                        estimated_output_tokens = len(raw_text) // 4

                        # Calculate cost
                        total_cost = (estimated_input_tokens * self.cost_per_input_token) + (
                                    estimated_output_tokens * self.cost_per_output_token)

                        # Log cost
                        logger.info(f"Mistral estimated {estimated_input_tokens} input tokens, {estimated_output_tokens} output tokens. Estimated cost: ${total_cost:.6f}")

                        # Clean and validate JSON response
                        cleaned_json = self._clean_and_validate_json(raw_text)
                        if cleaned_json:
                            return cleaned_json
                        else:
                            logger.error("No valid JSON found in the model response.")
                            return None
                    else:
                        logger.error("No output found in the model response.")
                        return None

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON from response: {e}")
                    return None

            except ClientError as e:
                logger.error(f"Failed to query Mistral model via Bedrock: {e}")

            attempt += 1
            if attempt < max_retries:
                logger.info(f"Retrying... (attempt {attempt + 1})")
                time.sleep(retry_delay)

        logger.error("Max retries reached. Failed to get a valid JSON response.")
        return None

    def _clean_and_validate_json(self, response_text):
        """
        Clean and validate JSON string from response text.
        """
        try:
            # Locate JSON in the text
            start_index = response_text.find('{')
            end_index = response_text.rfind('}') + 1
            if start_index == -1 or end_index == -1:
                logger.error("No JSON-like structure found in the response text.")
                return None

            # Extract potential JSON
            json_string = response_text[start_index:end_index].strip()

            # Clean the JSON (remove trailing commas, fix common issues)
            cleaned_string = self._remove_trailing_commas(json_string)

            # Parse and return JSON
            parsed_json = json.loads(cleaned_string)
            logger.info(f"Validated JSON: {parsed_json}")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in response: {e}")
            return None

    def _remove_trailing_commas(self, json_string):
        """
        Remove trailing commas from JSON-like strings to make them parseable.
        """
        import re
        # Remove trailing commas before closing braces/brackets
        cleaned_string = re.sub(r',\s*([\]}])', r'\1', json_string)
        return cleaned_string
