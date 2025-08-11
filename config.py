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
    
    # Hugging Face API
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"
    HUGGINGFACE_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
    
    # Terms and Conditions
    TERMS_CONDITIONS = '''
    **Terms and Conditions for ClauseWise Legal Document Analyzer**
    
    1. **Purpose**: This tool is designed for educational and informational purposes only.
    
    2. **No Legal Advice**: The analysis provided does not constitute legal advice. Always consult with a qualified attorney for legal matters.
    
    3. **Data Privacy**: Your uploaded documents are processed using IBM Watson and Hugging Face services and are handled according to their privacy policies.
    
    4. **Accuracy**: While we strive for accuracy using advanced AI, the analysis may contain errors. Please review all results carefully.
    
    5. **API Usage**: This tool uses IBM Watson and Hugging Face AI services for enhanced analysis.
    
    6. **Limitation of Liability**: We are not responsible for any decisions made based on the analysis provided by this tool.
    
    7. **Acceptance**: By using this service, you acknowledge that you have read, understood, and agree to these terms.
    '''