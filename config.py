import os

# Configuration settings
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
    WATSON_NLU_API_KEY = "dZKePvQ2q6KRTR1vk-o-RH6KnDTxFotOhBDp4c461RWL"
    WATSON_NLU_URL = "https://api.jp-tok.natural-language-understanding.watson.cloud.ibm.com/instances/c30bc15f-30ab-46d4-84db-0c74789e73b1"
    
    # IBM Watson Assistant
    WATSON_ASSISTANT_API_KEY = "cQr6ZJKZE7yj05yJ_rKhVcpn0EdtFtU06-laRKE2mHXu"
    WATSON_ASSISTANT_URL = "https://api.jp-tok.assistant.watson.cloud.ibm.com/instances/1b921179-9d40-407a-a604-0e8681b0176f"
    
    # IBM Granite Model (watsonx.ai)
    WATSONX_API_KEY = "0mKdImTNgmk9fWYsbWl6DbAZPoBrQ5cwDk-Rlcyg14a7"
    WATSONX_PROJECT_ID = "44c1fe48-953f-45e1-a043-806310d7365a"
    WATSONX_URL = "https://eu-de.ml.cloud.ibm.com"
    GRANITE_MODEL_ID = "ibm/granite-13b-instruct-v2"
    
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