import pytest
from app.llm_mistral import MistralBedrockAPI

@pytest.fixture
def mistral_instance():
    return MistralBedrockAPI()

def test_clean_and_validate_json_valid(mistral_instance):
    valid_json = '{"key": "value"}'
    assert mistral_instance._clean_and_validate_json(valid_json) == {"key": "value"}

def test_clean_and_validate_json_invalid(mistral_instance):
    invalid_json = '{"key": "value",}'
    cleaned_json = mistral_instance._remove_trailing_commas(invalid_json)
    assert mistral_instance._clean_and_validate_json(cleaned_json) == {"key": "value"}
