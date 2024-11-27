import os
import json
import logging
import importlib
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Environment variables
ocr_type = os.getenv('OCR_TYPE', 'textract').lower()  # Default to 'textract'
llm_type = os.getenv('LLM_TYPE', 'claude').lower()    # Default to 'claude'

# Dynamic module loading based on environment variables
# Dynamic module loading based on environment variables
if ocr_type == 'textract':
    from .s3_and_ocr_textract import TextractOCR as OCR
elif ocr_type == 'google':
    from .ocr_google import GoogleOCR as OCR
else:
    raise ValueError(f"Unsupported OCR_TYPE: {ocr_type}")

if llm_type == 'gpt4':
    from .llm_gpt4 import GPT4LLM as LLM
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found in environment variables")
    llm_instance = LLM(api_key)  # Correctly pass api_key
elif llm_type == 'claude':
    from .llm_claude import ClaudeBedrockAPI as LLM
    llm_instance = LLM()  # No api_key needed
elif llm_type == 'mistral':
    from .llm_mistral import MistralBedrockAPI as LLM
    llm_instance = LLM()  # No api_key needed
else:
    raise ValueError(f"Unsupported LLM_TYPE: {llm_type}")

# Create instances of the selected OCR class
ocr_instance = OCR()  # Initialize the OCR service

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    try:
        # File validation
        if 'file' not in request.files:
            return jsonify({"error": "No file part", "error_code": 201}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file", "error_code": 201}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "File is not a PDF", "error_code": 101}), 400

        # Read file as raw bytes
        file_bytes = file.read()

        # Get questions data from request
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

        # Use the selected OCR service to extract text and confidence score from the PDF
        extracted_text, average_confidence_score = ocr_instance.extract_text_from_pdf(file_bytes)
        if not extracted_text:
            return jsonify({"error": "No text extracted from the document", "error_code": 103}), 500

        # Dynamically call the appropriate method based on LLM_TYPE
        if llm_type == 'claude':
            llm_response = llm_instance.query_claude(extracted_text, questions)
        elif llm_type == 'mistral':
            llm_response = llm_instance.query_mistral(extracted_text, questions)
        elif llm_type == 'gpt4':
            llm_response = llm_instance.query_gpt4(extracted_text, questions)
        else:
            return jsonify({"error": f"Unsupported LLM_TYPE: {llm_type}", "error_code": 107}), 400

        # Include the confidence score in the response payload
        
        llm_response.update({"ocr_confidence_score": average_confidence_score})
        response_payload = llm_response

        return jsonify(response_payload), 200

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
        'app_name': "OCR and LLM Selection API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
