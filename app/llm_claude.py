import os
import json
import logging
import time
import boto3
import re
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeBedrockAPI:
    def __init__(self):
        self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        self.region_name = os.getenv('AWS_REGION', 'eu-west-3')
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region_name)

        # Cost per token (Claude 3 Sonnet pricing)
        self.cost_per_input_token = 0.0025 / 1000  # $0.0025 per 1K input tokens
        self.cost_per_output_token = 0.015 / 1000  # $0.015 per 1K output tokens

    def validate_json(self, response_text):
        """Validate and parse JSON response, removing markdown if necessary."""
        try:
            # Strip markdown formatting safely
            response_text = re.sub(r"^```json|```$", "", response_text.strip(), flags=re.MULTILINE).strip()

            # Attempt to parse the cleaned JSON
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None

    def query_claude(self, extracted_text, questions, prefilled_response=None, max_retries=3, retry_delay=2):
        """Query Claude with extracted text and questions, logging the cost."""
        question_instructions = ", ".join([f'"{q["field_name"]}": "{q["question"]}"' for q in questions])

        system_prompt = "You are given the extracted text from a document. Answer the questions in JSON format."

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
        The document contains details such as:
        - Certificates (name, type, auditor, issue date, validity)
        - Shipments (shipment number, date, weight, invoices)

        Instructions:
        - Use JSON format only.
        - Dates must be YYYY-MM-DD.
        - Use NULL for missing values.

        Extracted text:
        {extracted_text}

        Answer the following:
        {{
        {question_instructions}
        }}

        Expected JSON:
        ```json
        {json_prefill}
        ```
        """

        messages = [{"role": "user", "content": user_message_content}]

        if prefilled_response:
            messages.append({"role": "assistant", "content": prefilled_response})

        body = json.dumps({
            "anthropic_version": 'bedrock-2023-05-31',
            "system": system_prompt,
            "messages": messages,
            "max_tokens": 8000
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
                logger.info(f"Claude Full Response: {response_body}")

                # Extract token usage safely
                input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
                output_tokens = response_body.get("usage", {}).get("output_tokens", 0)

                # Calculate cost based on Sonnet pricing
                total_cost = (input_tokens * self.cost_per_input_token) + (output_tokens * self.cost_per_output_token)

                # Log token usage and cost
                logger.info(f"Tokens Used: Input={input_tokens}, Output={output_tokens} | Estimated Cost: ${total_cost:.6f}")

                # Extract and validate JSON response
                result = response_body.get('content', [{}])[0].get('text', '').strip()
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
