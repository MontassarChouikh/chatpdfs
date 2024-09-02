import boto3
import os
from pdf2image import convert_from_path
from botocore.exceptions import ClientError

# Initialize AWS clients for S3 and Textract
s3_client = boto3.client('s3')
textract_client = boto3.client('textract')

def convert_pdf_to_images(pdf_file, output_folder='output_images'):
    """
    Convert each page of a PDF into an image using pdf2image.
    """
    try:
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Convert the PDF into images
        images = convert_from_path(pdf_file)
        
        image_filenames = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"page_{i+1}.png")
            image.save(image_path, 'PNG')
            image_filenames.append(image_path)
            print(f"Page {i+1} saved as image: {image_path}")

        return image_filenames
    except Exception as e:
        print(f"An error occurred during PDF to image conversion: {e}")
        return []

def upload_image_to_s3(s3_bucket, image_file):
    """
    Upload an image to S3.
    """
    try:
        s3_client.upload_file(image_file, s3_bucket, image_file)
        print(f"File {image_file} successfully uploaded to bucket {s3_bucket}")
        return True
    except ClientError as e:
        print(f"Failed to upload file: {e}")
        return False

def extract_text_from_image(s3_bucket, image_file):
    """
    Extract text from an image using AWS Textract.
    """
    try:
        # Synchronously detect text from the image using Textract
        response = textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': s3_bucket,
                    'Name': image_file
                }
            }
        )

        # Extract the text
        extracted_text = ""
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                extracted_text += block['Text'] + "\n"

        print(f"Extracted text from {image_file}:\n{extracted_text}")
        return extracted_text

    except ClientError as e:
        print(f"Failed to extract text using Textract: {e}")
        return ""

def main():
    # Configurer le nom du bucket S3 et le fichier PDF
    s3_bucket = 'ai-bucket'  # Remplacez par votre bucket S3
    pdf_file = 'chanel2.pdf'  # Remplacez par le nom de votre fichier

    # Convertir le PDF en images
    print("Starting PDF to image conversion...")
    image_files = convert_pdf_to_images(pdf_file)

    if not image_files:
        print("No images were generated. Exiting.")
        return

    # Upload each image to S3 and run Textract on each one
    for image_file in image_files:
        uploaded = upload_image_to_s3(s3_bucket, image_file)
        if uploaded:
            # If the image was uploaded successfully, extract text using Textract
            extract_text_from_image(s3_bucket, image_file)

if __name__ == "__main__":
    main()
