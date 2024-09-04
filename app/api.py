import io
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from .core_claude import extract_text_from_pdf, query_claude_via_bedrock
import json
import logging
import re

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def clean_and_format_json(response_text):
    if isinstance(response_text, dict):
        return response_text

    cleaned_response = response_text.strip().replace("\\n", "").replace("\\", "")
    cleaned_response = re.sub(r",\s*}", "}", cleaned_response)
    cleaned_response = re.sub(r",\s*]", "]", cleaned_response)

    if not cleaned_response.startswith("{") and not cleaned_response.startswith("["):
        cleaned_response = f"[{cleaned_response}]"

    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}")
        return None

@app.route('/process-pdf', methods=['POST'])  # Updated route name
def process_pdf():  # Updated function name
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part", "error_code": 201}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file", "error_code": 201}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "File is not a PDF", "error_code": 101}), 400

        # Convert the file to raw bytes for pdf2image
        file_bytes = file.read()  # Read the file as raw bytes

        questions_data = request.form.get('questions')
        if not questions_data:
            return jsonify({"error": "No questions data provided", "error_code": 106}), 400

        try:
            questions = json.loads(questions_data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format in questions data", "error_code": 104}), 400

        # Prepend a predefined text to each question
        for question in questions:
            question['question'] = f"Who is the {question['question']}"

        # Pass the raw bytes to the `extract_text_from_pdf` function
        extracted_text, average_confidence_score = extract_text_from_pdf('ai-bucket', file_bytes)
        if not extracted_text:
            return jsonify({"error": "No text extracted from the document", "error_code": 103}), 500

        # Update to query Claude instead of Mistral
        raw_response = query_claude_via_bedrock(extracted_text, questions)

        if isinstance(raw_response, str):
            cleaned_response = clean_and_format_json(raw_response)
        else:
            cleaned_response = raw_response

        if cleaned_response:
            # Add confidence score after each answer field
            if isinstance(cleaned_response, dict):
                cleaned_response['average_confidence_score'] = average_confidence_score

            # Include the average confidence score in the response
            return jsonify(cleaned_response), 200
        else:
            return jsonify({"error": "Failed to process model response", "raw_response": raw_response}), 500

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e), "error_code": 500}), 500


# Swagger UI setup
SWAGGER_URL = '/apidocs'
API_URL = '/static/swagger.yaml'  # Adjust to the path of your Swagger YAML file

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Textract & Claude API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
