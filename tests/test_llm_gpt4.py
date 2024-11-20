import pytest
from unittest.mock import MagicMock, patch
from app.llm_gpt4 import GPT4LLM

@pytest.fixture
def gpt4_instance():
    """
    Fixture to create an instance of GPT4LLM.
    """
    return GPT4LLM(api_key="test-api-key")


def test_query_gpt4_success(mocker, gpt4_instance):
    """
    Test that query_gpt4 returns structured JSON when GPT-4 responds successfully.
    """
    # Mock the ChatPromptTemplate.from_template
    mock_prompt_template = mocker.patch("app.llm_gpt4.ChatPromptTemplate.from_template")

    # Mock the chain behavior
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "CertificateName": "Transaction Certificate",
        "CertificateType": "GOTS",
        "CertificateAuditor": "Auditor Name",
        "CertificateIssueDate": "2024-01-01",
        "CertificateValidityStartDate": "2024-01-01",
        "CertificateValidityEndDate": "2025-01-01",
        "shipments": [
            {
                "ShipmentNumber": 1,
                "ShipmentDate": "2024-01-02",
                "GrossShippingWeight": 100.0,
                "InvoiceReferences": "INV123"
            }
        ]
    }

    # Set the mock chain to the return value of the prompt template
    mock_prompt_template.return_value.__or__.return_value.__or__.return_value = mock_chain

    # Test inputs
    extracted_text = "Sample extracted text"
    questions = [{"field_name": "name", "question": "What is the certificate name?"}]

    # Call the method
    result = gpt4_instance.query_gpt4(extracted_text, questions)

    # Assert the result matches the mocked return
    assert result["CertificateName"] == "Transaction Certificate"
    assert result["CertificateType"] == "GOTS"
    assert result["CertificateIssueDate"] == "2024-01-01"

def test_query_gpt4_error(mocker, gpt4_instance):
    """
    Test that query_gpt4 handles errors gracefully.
    """
    # Mock the ChatPromptTemplate.from_template
    mock_prompt_template = mocker.patch("app.llm_gpt4.ChatPromptTemplate.from_template")

    # Mock the chain behavior with an exception
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("Simulated error")

    # Set the mock chain to the return value of the prompt template
    mock_prompt_template.return_value.__or__.return_value.__or__.return_value = mock_chain

    # Test inputs
    extracted_text = "Sample extracted text"
    questions = [{"field_name": "name", "question": "What is the certificate name?"}]

    # Call the method
    result = gpt4_instance.query_gpt4(extracted_text, questions)

    # Assert the result is None due to error
    assert result is None
