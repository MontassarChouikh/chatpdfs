import pytest
from app.llm_claude import ClaudeBedrockAPI

@pytest.fixture
def claude_instance():
    """
    Fixture to initialize a ClaudeBedrockAPI instance.
    """
    return ClaudeBedrockAPI()

def test_validate_json_valid(claude_instance):
    """
    Test that validate_json correctly parses valid JSON.
    """
    valid_json = '{"key": "value"}'
    assert claude_instance.validate_json(valid_json) == {"key": "value"}

def test_validate_json_invalid(claude_instance):
    """
    Test that validate_json returns None for invalid JSON.
    """
    invalid_json = '{"key": "value"'
    assert claude_instance.validate_json(invalid_json) is None

def test_query_claude_success(mocker, claude_instance):
    """
    Test query_claude to ensure it processes correctly when Claude returns valid JSON.
    """
    # Mock the Bedrock client invocation
    mock_bedrock_client = mocker.patch.object(claude_instance.bedrock_client, "invoke_model")
    mock_bedrock_client.return_value = {
        'body': mocker.Mock(read=lambda: b'{"content": [{"text": "{\\"key\\": \\"value\\"}"}]}')
    }

    # Test inputs
    extracted_text = "Sample text"
    questions = [{"field_name": "name", "question": "What is the name?"}]

    # Call the function
    response = claude_instance.query_claude(extracted_text, questions)

    # Assert response
    assert response == {"key": "value"}

def test_query_claude_invalid_json(mocker, claude_instance):
    """
    Test query_claude to ensure it handles invalid JSON gracefully.
    """
    # Mock the Bedrock client invocation with invalid JSON response
    mock_bedrock_client = mocker.patch.object(claude_instance.bedrock_client, "invoke_model")
    mock_bedrock_client.return_value = {
        'body': mocker.Mock(read=lambda: b'{"content": [{"text": "{\\"key\\"}"}]}')
    }

    # Test inputs
    extracted_text = "Sample text"
    questions = [{"field_name": "name", "question": "What is the name?"}]

    # Call the function
    response = claude_instance.query_claude(extracted_text, questions)

    # Assert response is None due to invalid JSON
    assert response is None


def test_query_claude_client_error(mocker, claude_instance):
    """
    Test query_claude to handle ClientError exceptions from Bedrock.
    """
    # Mock the Bedrock client to raise an exception
    mock_bedrock_client = mocker.patch.object(claude_instance.bedrock_client, "invoke_model")
    mock_bedrock_client.side_effect = Exception("Simulated Bedrock failure")

    # Test inputs
    extracted_text = "Sample text"
    questions = [{"field_name": "name", "question": "What is the name?"}]

    # Call the function
    response = claude_instance.query_claude(extracted_text, questions)

    # Assert response is None
    assert response is None

def test_query_claude_retries(mocker, claude_instance):
    """
    Test query_claude to ensure it retries on failure.
    """
    # Mock the Bedrock client to fail for the first two attempts
    mock_bedrock_client = mocker.patch.object(claude_instance.bedrock_client, "invoke_model")
    mock_bedrock_client.side_effect = [
        Exception("Simulated failure 1"),
        Exception("Simulated failure 2"),
        {
            'body': mocker.Mock(read=lambda: b'{"content": [{"text": "{\\"key\\": \\"value\\"}"}]}')
        }
    ]

    # Test inputs
    extracted_text = "Sample text"
    questions = [{"field_name": "name", "question": "What is the name?"}]

    # Call the function
    response = claude_instance.query_claude(extracted_text, questions)

    # Assert the final successful response
    assert response == {"key": "value"}

    # Verify retry attempts
    assert mock_bedrock_client.call_count == 3
