import pytest
from unittest.mock import MagicMock
from app.s3_and_ocr_textract import TextractOCR


@pytest.fixture
def textract_instance():
    """
    Fixture to create an instance of TextractOCR with a mocked region.
    """
    return TextractOCR(region_name='eu-west-1')


def test_convert_pdf_to_images(textract_instance, mocker):
    """
    Test convert_pdf_to_images to ensure it processes PDF files correctly.
    """
    # Mock the pdf2image convert_from_bytes method
    mock_convert = mocker.patch("app.s3_and_ocr_textract.convert_from_bytes", return_value=["image1", "image2"])

    pdf_file = b"fake-pdf-content"
    images = textract_instance.convert_pdf_to_images(pdf_file)

    mock_convert.assert_called_once_with(pdf_file)  # Ensure the function was called
    assert images == ["image1", "image2"]


def test_upload_images_to_s3(textract_instance, mocker):
    """
    Test upload_images_to_s3 to ensure it uploads images and returns paths.
    """
    mock_s3_client = mocker.patch.object(textract_instance.s3_client, "upload_file")

    # Mock image save method
    mock_image = MagicMock()
    mock_image.save = MagicMock()

    images = [mock_image, mock_image]
    uploaded_paths = textract_instance.upload_images_to_s3(images)

    # Check that upload_file was called for each image
    assert mock_s3_client.call_count == len(images)
    assert len(uploaded_paths) == len(images)
    assert all(path.startswith("pdf_image_") for path in uploaded_paths)


def test_extract_text_and_confidence(textract_instance, mocker):
    """
    Test extract_text_and_confidence to ensure it processes images and calculates confidence.
    """
    mock_textract_client = mocker.patch.object(textract_instance.textract_client, "analyze_document")
    mock_textract_client.return_value = {
        "Blocks": [
            {"BlockType": "LINE", "Text": "Test text", "Confidence": 99.0},
            {"BlockType": "LINE", "Text": "Another line", "Confidence": 98.0}
        ]
    }

    image_paths = ["s3://bucket/image1.png"]
    text, confidence = textract_instance.extract_text_and_confidence(image_paths)

    mock_textract_client.assert_called_once_with(
        Document={'S3Object': {'Bucket': textract_instance.s3_bucket, 'Name': image_paths[0]}},
        FeatureTypes=["TABLES", "FORMS"]
    )
    assert text == "Test text Another line"
    assert confidence == 98.5  # Average of 99.0 and 98.0


def test_extract_text_from_pdf(textract_instance, mocker):
    """
    Test extract_text_from_pdf to ensure the complete process works correctly.
    """
    # Mock individual methods
    mock_convert = mocker.patch.object(textract_instance, "convert_pdf_to_images", return_value=["image1", "image2"])
    mock_upload = mocker.patch.object(textract_instance, "upload_images_to_s3", return_value=["s3://bucket/image1", "s3://bucket/image2"])
    mock_extract = mocker.patch.object(textract_instance, "extract_text_and_confidence", return_value=("Extracted text", 90.0))

    pdf_file = b"fake-pdf-content"
    text, confidence = textract_instance.extract_text_from_pdf(pdf_file)

    # Ensure methods were called in sequence
    mock_convert.assert_called_once_with(pdf_file)
    mock_upload.assert_called_once_with(["image1", "image2"])
    mock_extract.assert_called_once_with(["s3://bucket/image1", "s3://bucket/image2"])

    assert text == "Extracted text"
    assert confidence == 90.0
