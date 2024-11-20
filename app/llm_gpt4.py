import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4LLM:
    def __init__(self, api_key):
        """
        Initialize the OpenAI GPT-4 LLM via LangChain with the provided API key.
        """
        self.model = ChatOpenAI(
            api_key=api_key, 
            model="gpt-4o",  # Using GPT-4 via LangChain
            temperature=0.5,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def query_gpt4(self, extracted_text, questions):
        """
        Query GPT-4 using LangChain's pipeline, ensuring JSON structured output.
        
        Args:
            extracted_text (str): The text extracted from the document.
            questions (list): List of questions to ask based on the text.

        Returns:
            dict: A JSON object with structured answers.
        """
        # Format questions into a prompt
        question_instructions = ", ".join([f'"{q["field_name"]}": "{q["question"]}"' for q in questions])

        # Create a prompt template including an example
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are given the extracted text from a document. 
            Please answer the following questions in a JSON format based on the provided text.

            Example answer format:
            {{
                "CertificateName": "Transaction Certificate",
                "CertificateType": "Transaction Certificate Type",
                "CertificateAuditor": "Auditor Name",
                "CertificateIssueDate": "YYYY-MM-DD",
                "CertificateValidityStartDate": "YYYY-MM-DD",
                "CertificateValidityEndDate": "YYYY-MM-DD",
                "shipments": [
                    {{
                        "ShipmentNumber": 1,
                        "ShipmentDate": "YYYY-MM-DD",
                        "GrossShippingWeight": 100.0,
                        "InvoiceReferences": "Invoice Number"
                    }}
                ]
            }}

            Extracted text: {extracted_text}

            Questions: {question_instructions}
            """
        )

        # Set up the chain with the model and parser
        parser = SimpleJsonOutputParser()
        chain = prompt_template | self.model | parser

        try:
            # Prepare input for the chain
            input_data = {
                "extracted_text": extracted_text,
                "question_instructions": question_instructions
            }

            # Run the chain and get the structured JSON output
            result = chain.invoke(input_data)
            logger.info(f"Response from GPT-4: {result}")
            return result

        except Exception as e:
            logger.error(f"Error during query: {e}")
            return None
