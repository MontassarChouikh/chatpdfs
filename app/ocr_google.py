import os
import json
import logging
import tempfile
from google.cloud import documentai_v1 as documentai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleOCR:
    def __init__(self):
        # Load credentials from environment variable
        google_credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        
        if google_credentials_json:
            # Write the JSON credentials to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
                temp_file.write(google_credentials_json)
                self.temp_file_path = temp_file.name
            
            # Set the environment variable for Google Cloud credentials
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.temp_file_path
        else:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set")
        
        # Initialize the Document AI client
        self.documentai_client = documentai.DocumentProcessorServiceClient()

    def extract_text_from_pdf(self, image_file):
        """
        Extract text and calculate confidence scores from a PDF using Google Document AI.
        """
        try:
            # Replace placeholders with actual values
            processor_id = '78e04735f550c004'
            project_id = 'analyse-pdf-423009'  # Replace with your Google Cloud project ID
            location = 'us'

            # Construct the full processor resource name
            processor_name = f'projects/{project_id}/locations/{location}/processors/{processor_id}'

            # Construct the request
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=documentai.RawDocument(
                    content=image_file,  # Pass raw file data
                    mime_type='application/pdf'
                )
            )

            # Process the document
            result = self.documentai_client.process_document(request=request)
            document = result.document
            document_text = document.text

            # Log the full response for debugging
            logger.debug(f"Full Document AI response: {document}")

            # Extract confidence scores from entities
            confidence_scores = []
            if hasattr(document, 'entities') and document.entities:
                confidence_scores.extend(entity.confidence for entity in document.entities)

            # Fallback: Extract confidence from blocks if entities are not present
            if not confidence_scores and hasattr(document, 'pages'):
                for page in document.pages:
                    for block in page.blocks:
                        confidence_scores.append(block.layout.confidence)

            # Calculate the average confidence score
            average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            logger.info(f"Extracted text with Google Document AI: {document_text}")
            logger.info(f"Average confidence score: {average_confidence:.2f}")

            return document_text, average_confidence
        except Exception as e:
            logger.error(f"Failed to extract text with Google Document AI: {e}")
            return "", 0.0
