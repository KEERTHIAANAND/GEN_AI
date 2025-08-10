import PyPDF2
import docx
import re
from typing import Dict, List
import nltk
from nltk.tokenize import sent_tokenize
from config import Config

# IBM Watson imports
try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False
    print("Watson SDK not available, using fallback methods")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class DocumentProcessor:
    def __init__(self):
        self.watson_nlu = None
        
        # Initialize Watson NLU for enhanced entity extraction
        if WATSON_AVAILABLE and Config.WATSON_NLU_API_KEY:
            try:
                authenticator = IAMAuthenticator(Config.WATSON_NLU_API_KEY)
                self.watson_nlu = NaturalLanguageUnderstandingV1(
                    version='2022-04-07',
                    authenticator=authenticator
                )
                self.watson_nlu.set_service_url(Config.WATSON_NLU_URL)
                print("Watson NLU initialized successfully in DocumentProcessor")
            except Exception as e:
                print(f"Watson NLU initialization failed in DocumentProcessor: {e}")
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {str(e)}")
    
    def extract_text_from_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                file.seek(0)
                return file.read().decode('latin-1')
            except Exception as e:
                raise Exception(f"Error extracting TXT text: {str(e)}")
        except Exception as e:
            raise Exception(f"Error extracting TXT text: {str(e)}")
    
    def extract_text(self, file, file_type: str) -> str:
        """Extract text based on file type"""
        if file_type == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_type == 'docx':
            return self.extract_text_from_docx(file)
        elif file_type == 'txt':
            return self.extract_text_from_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        try:
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text)
            # Remove problematic characters but keep basic punctuation
            text = re.sub(r'[^\w\s.,;:!?()\-\[\]"\']', ' ', text)
            # Fix multiple spaces
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception:
            # If regex fails, just clean whitespace
            return ' '.join(text.split())
    
    def extract_clauses(self, text: str) -> List[str]:
        """Extract individual clauses from text"""
        try:
            clauses = []
            
            # Try different clause splitting methods
            # Method 1: Split by numbered clauses
            numbered_pattern = r'\n\s*\d+\.\s+'
            parts = re.split(numbered_pattern, text, flags=re.IGNORECASE)
            if len(parts) > 2:
                clauses = [part.strip() for part in parts[1:] if part.strip() and len(part.strip()) > 50]
            
            # Method 2: If no numbered clauses, try paragraphs
            if not clauses:
                paragraphs = text.split('\n\n')
                clauses = [p.strip() for p in paragraphs if len(p.strip()) > 50]
            
            # Method 3: If still no clauses, split by sentences
            if not clauses:
                sentences = sent_tokenize(text)
                current_clause = ""
                for sentence in sentences:
                    current_clause += sentence + " "
                    if len(current_clause) > 100:
                        clauses.append(current_clause.strip())
                        current_clause = ""
                if current_clause.strip():
                    clauses.append(current_clause.strip())
            
            return clauses[:10]  # Limit to first 10 clauses
            
        except Exception as e:
            print(f"Error extracting clauses: {e}")
            # Emergency fallback
            sentences = text.split('. ')
            return [s.strip() + '.' for s in sentences if len(s.strip()) > 50][:5]
    
    def extract_entities_with_watson(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using Watson NLU"""
        if not self.watson_nlu:
            return self.extract_entities_fallback(text)
        
        try:
            # Limit text size for API
            analysis_text = text[:4000] if len(text) > 4000 else text
            
            response = self.watson_nlu.analyze(
                text=analysis_text,
                features=Features(
                    entities=EntitiesOptions(limit=25),
                    keywords=KeywordsOptions(limit=20)
                )
            ).get_result()
            
            entities = {
                'parties': [],
                'dates': [],
                'monetary_values': [],
                'obligations': [],
                'legal_terms': []
            }
            
            # Extract entities from Watson response
            watson_entities = response.get('entities', [])
            for entity in watson_entities:
                entity_type = entity.get('type', '').lower()
                entity_text = entity.get('text', '').strip()
                confidence = entity.get('confidence', 0)
                
                if confidence > 0.6:  # Higher confidence threshold
                    if entity_type in ['person', 'organization', 'company']:
                        entities['parties'].append(entity_text)
                    elif entity_type in ['date', 'datetime', 'time']:
                        entities['dates'].append(entity_text)
                    elif entity_type in ['money', 'currency', 'quantity']:
                        entities['monetary_values'].append(entity_text)
            
            # Extract keywords as legal terms
            keywords = response.get('keywords', [])
            for keyword in keywords:
                kw_text = keyword.get('text', '')
                kw_relevance = keyword.get('relevance', 0)
                if kw_relevance > 0.6:
                    entities['legal_terms'].append(kw_text)
            
            # Extract obligations
            try:
                sentences = sent_tokenize(text)
                obligation_words = ['shall', 'must', 'agrees', 'obligated', 'required', 'responsible']
                for sentence in sentences:
                    if any(word in sentence.lower() for word in obligation_words) and len(sentence) < 150:
                        entities['obligations'].append(sentence.strip())
                        if len(entities['obligations']) >= 3:
                            break
            except Exception:
                pass
            
            # Remove duplicates and limit results
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))[:5]
            
            return entities
            
        except Exception as e:
            print(f"Watson entity extraction failed: {e}")
            return self.extract_entities_fallback(text)
    
    def extract_entities_fallback(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity extraction using regex patterns"""
        entities = {
            'parties': [],
            'dates': [],
            'monetary_values': [],
            'obligations': [],
            'legal_terms': []
        }
        
        try:
            # Extract dates
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['dates'].extend(matches)
            
            # Extract monetary values
            money_patterns = [
                r'\$[\d,]+(?:\.\d{2})?',
                r'USD\s*[\d,]+',
                r'[\d,]+\s*dollars?'
            ]
            
            for pattern in money_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['monetary_values'].extend(matches)
            
            # Extract common legal terms
            legal_terms = ['agreement', 'contract', 'party', 'parties', 'obligation', 'liability', 
                          'termination', 'breach', 'confidential', 'proprietary', 'indemnification']
            for term in legal_terms:
                if term.lower() in text.lower():
                    entities['legal_terms'].append(term.title())
            
            # Extract potential party names
            party_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|LLC|Corp|Company|Ltd))?\b'
            potential_parties = re.findall(party_pattern, text)
            entities['parties'] = [p for p in potential_parties if len(p.split()) <= 3][:3]
            
            # Extract obligations
            try:
                sentences = sent_tokenize(text)
                obligation_words = ['shall', 'must', 'agrees', 'obligated', 'required']
                for sentence in sentences:
                    if any(word in sentence.lower() for word in obligation_words) and len(sentence) < 150:
                        entities['obligations'].append(sentence.strip())
                        if len(entities['obligations']) >= 3:
                            break
            except Exception:
                pass
            
            # Remove duplicates and limit results
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))[:5]
            
            # Ensure we have some content
            if not any(entities.values()):
                entities = {
                    'parties': ['Document analysis completed'],
                    'dates': ['See full document'],
                    'monetary_values': ['See full document'],
                    'obligations': ['See full document'],
                    'legal_terms': ['Legal document detected']
                }
            
            return entities
            
        except Exception as e:
            print(f"Fallback entity extraction failed: {e}")
            return {
                'parties': ['Analysis completed'],
                'dates': ['See document'],
                'monetary_values': ['See document'],
                'obligations': ['See document'],
                'legal_terms': ['Legal content detected']
            }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Main entity extraction method"""
        if self.watson_nlu:
            return self.extract_entities_with_watson(text)
        else:
            return self.extract_entities_fallback(text)