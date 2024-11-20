import pytest
from app.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_process_pdf_success(client, mocker):
    mock_ocr = mocker.patch("app.api.OCR.extract_text_from_pdf", return_value=("Sample text", 0.95))
    mock_llm = mocker.patch("app.api.LLM.query_claude", return_value={"key": "value"})

    data = {
        "questions": '[{"field_name": "name", "question": "What is the name?"}]',
    }
    with open("1.pdf", "rb") as pdf_file:
        data["file"] = pdf_file
        response = client.post("/process-pdf", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    json_response = response.get_json()
    assert json_response["llm_response"] == {"key": "value"}
    assert json_response["ocr_confidence_score"] == 0.95
