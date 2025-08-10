import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = ['pdf', 'docx', 'txt']
    
    # Document types for classification
    DOCUMENT_TYPES = [
        'NDA (Non-Disclosure Agreement)',
        'Employment Contract',
        'Service Agreement',
        'Lease Agreement',
        'Purchase Agreement',
        'Partnership Agreement',
        'License Agreement',
        'Other Legal Document'
    ]
    
    # IBM Watson Natural Language Understanding
    WATSON_NLU_API_KEY = os.getenv('WATSON_NLU_API_KEY')
    WATSON_NLU_URL = os.getenv('WATSON_NLU_URL')
    
    # IBM Watson Assistant
    WATSON_ASSISTANT_API_KEY = os.getenv('WATSON_ASSISTANT_API_KEY')
    WATSON_ASSISTANT_URL = os.getenv('WATSON_ASSISTANT_URL')
    
    # IBM Granite Model (watsonx.ai)
    WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
    WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
    WATSONX_URL = os.getenv('WATSONX_URL')
    GRANITE_MODEL_ID = os.getenv('GRANITE_MODEL_ID', 'ibm/granite-13b-instruct-v2')
    
    # Terms and Conditions
    TERMS_CONDITIONS = '''
    **Terms and Conditions for ClauseWise Legal Document Analyzer**
    
    1. **Purpose**: This tool is designed for educational and informational purposes only.
    
    2. **No Legal Advice**: The analysis provided does not constitute legal advice. Always consult with a qualified attorney for legal matters.
    
    3. **Data Privacy**: Your uploaded documents are processed using IBM Watson services and are handled according to IBM's privacy policies.
    
    4. **Accuracy**: While we strive for accuracy using advanced AI, the analysis may contain errors. Please review all results carefully.
    
    5. **API Usage**: This tool uses IBM Watson and Granite AI services for enhanced analysis.
    
    6. **Limitation of Liability**: We are not responsible for any decisions made based on the analysis provided by this tool.
    
    7. **Acceptance**: By using this service, you acknowledge that you have read, understood, and agree to these terms.
    '''