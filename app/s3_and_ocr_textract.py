import boto3
import logging
from pdf2image import convert_from_bytes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextractOCR:
    def __init__(self, region_name='eu-west-1'):
        self.textract_client = boto3.client('textract', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.s3_bucket = 'ai-bucket'  # Set your S3 bucket name here

    def convert_pdf_to_images(self, pdf_file):
        """
        Convert each page of a PDF (in-memory) into an image using pdf2image.
        """
        try:
            images = convert_from_bytes(pdf_file)
            return images
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []

    def upload_images_to_s3(self, images):
        """
        Uploads images to S3 and returns the list of file paths in S3.
        """
        image_paths = []
        try:
            for i, image in enumerate(images):
                image_path = f"pdf_image_{i+1}.png"
                image.save(f"/tmp/{image_path}", 'PNG')  # Save image temporarily
                self.s3_client.upload_file(Filename=f"/tmp/{image_path}", Bucket=self.s3_bucket, Key=image_path)
                image_paths.append(image_path)
                logger.info(f"Uploaded {image_path} to S3 bucket {self.s3_bucket}")
        except Exception as e:
            logger.error(f"Failed to upload images to S3: {e}")
        return image_paths

    def extract_text_and_confidence(self, image_paths):
        """
        Extract text from images stored in S3 using Textract and calculate confidence scores.
        """
        all_text = []
        all_confidence_scores = []
        try:
            for image_path in image_paths:
                response = self.textract_client.analyze_document(
                    Document={'S3Object': {'Bucket': self.s3_bucket, 'Name': image_path}},
                    FeatureTypes=["TABLES", "FORMS"])

                page_text = []
                page_confidence_scores = []
                for item in response["Blocks"]:
                    if item["BlockType"] == "LINE":
                        page_text.append(item["Text"])
                        page_confidence_scores.append(item["Confidence"])
            
                all_text.append(" ".join(page_text))  # Combine text of one page into one string
                all_confidence_scores.extend(page_confidence_scores)  # Collect all confidence scores
                logger.info(f"Extracted text from {image_path} with confidence.")

        except Exception as e:
            logger.error(f"Failed to extract text from images: {e}")
            return [], []

        average_confidence = sum(all_confidence_scores) / len(all_confidence_scores) if all_confidence_scores else 0
        return " ".join(all_text), average_confidence

    def extract_text_from_pdf(self, pdf_file):
        """
        Convert the PDF file to images, upload these images to S3, and extract text from them using Textract.
        """
        images = self.convert_pdf_to_images(pdf_file)
        if not images:
            logger.error("No images created from PDF.")
            return None, 0

        image_paths = self.upload_images_to_s3(images)
        if not image_paths:
            logger.error("No images uploaded to S3.")
            return None, 0

        extracted_text, average_confidence = self.extract_text_and_confidence(image_paths)
        # Print the extracted text and average confidence score
        logger.info(f"Extracted Text: {extracted_text}")
        logger.info(f"Average Confidence Score: {average_confidence:.2f}")

        return extracted_text, average_confidence
