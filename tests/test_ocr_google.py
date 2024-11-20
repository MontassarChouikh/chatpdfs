import pytest
from unittest.mock import patch, MagicMock
from app.ocr_google import GoogleOCR

@pytest.fixture
def google_ocr_instance():
    # Mock the credentials handling in __init__
    with patch("app.ocr_google.os.getenv") as mock_getenv, \
         patch("app.ocr_google.tempfile.NamedTemporaryFile") as mock_tempfile, \
         patch("google.auth.default") as mock_google_auth:
        
        # Mock environment variables
        def getenv_side_effect(key, default=None):
            if key == "GOOGLE_APPLICATION_CREDENTIALS_JSON":
                return '{"type": "service_account", "project_id": "mock_project"}'
            if key == "GOOGLE_API_USE_CLIENT_CERTIFICATE":
                return "false"  # Valid value for the environment variable
            return default

        mock_getenv.side_effect = getenv_side_effect

        # Mock a temporary file
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/mock_google_credentials.json"  # Fake path
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        # Mock Google authentication to bypass the need for a real file
        mock_google_auth.return_value = (MagicMock(), "mock-project")

        # Return the mocked GoogleOCR instance
        return GoogleOCR()

def test_extract_text_from_pdf(google_ocr_instance, mocker):
    # Mock the DocumentProcessorServiceClient
    mock_client = mocker.patch.object(google_ocr_instance, "documentai_client")
    
    # Set up the mock response for `process_document`
    mock_response = MagicMock()
    mock_response.document.text = "Extracted text"
    mock_response.document.entities = [MagicMock(confidence=0.9)]
    mock_response.document.pages = [MagicMock(blocks=[MagicMock(layout=MagicMock(confidence=0.9))])]
    mock_client.process_document.return_value = mock_response

    # Run the function under test
    text, confidence = google_ocr_instance.extract_text_from_pdf(b"pdf-data")

    # Assertions
    assert text == "Extracted text"
    assert confidence == 0.9
