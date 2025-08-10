import PyPDF2
import docx
import re
from typing import Dict, List, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from config import Config

# IBM Watson imports
try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DocumentProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.watson_nlu = None
        
        # Initialize Watson NLU for enhanced entity extraction
        if WATSON_AVAILABLE:
            try:
                authenticator = IAMAuthenticator(Config.WATSON_NLU_API_KEY)
                self.watson_nlu = NaturalLanguageUnderstandingV1(
                    version='2022-04-07',
                    authenticator=authenticator
                )
                self.watson_nlu.set_service_url(Config.WATSON_NLU_URL)
            except Exception as e:
                print(f"Watson NLU initialization failed in DocumentProcessor: {e}")
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {str(e)}")
    
    def extract_text_from_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
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
            text = re.sub(r'[^\w\s.,;:!?()\-\[\]"]', ' ', text)
            # Fix multiple spaces
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception:
            # If regex fails, just clean whitespace
            return ' '.join(text.split())
    
    def extract_clauses(self, text: str) -> List[str]:
        """Extract individual clauses from text"""
        try:
            # Split by common clause indicators - fixed regex patterns
            clause_patterns = [
                r'\n\s*\d+\.\s+',  # Numbered clauses (1. 2. 3.)
                r'\n\s*[A-Z]\)\s+',  # Lettered clauses (A) B) C))
                r'\n\s*Article\s+\d+',  # Articles
                r'\n\s*Section\s+\d+',  # Sections
                r'\n\s*Clause\s+\d+',  # Explicit clauses
            ]
            
            clauses = []
            current_text = text
            
            for pattern in clause_patterns:
                try:
                    parts = re.split(pattern, current_text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        clauses.extend([part.strip() for part in parts if part.strip() and len(part.strip()) > 20])
                        break
                except re.error:
                    continue
            
            if not clauses:
                # Fallback: split by paragraphs and double newlines
                paragraphs = text.replace('\n\n', '|||SPLIT|||').split('|||SPLIT|||')
                clauses = [p.strip() for p in paragraphs if len(p.strip()) > 50]
            
            # If still no clauses, split by sentences for very short documents
            if not clauses:
                sentences = sent_tokenize(text)
                clauses = [s.strip() for s in sentences if len(s.strip()) > 30]
            
            return clauses[:10]  # Limit to first 10 clauses
            
        except Exception as e:
            # Emergency fallback - just split by periods for sentences
            sentences = text.split('. ')
            return [s.strip() + '.' for s in sentences if len(s.strip()) > 30][:10]
    
    def extract_entities_with_watson(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using Watson NLU"""
        if not self.watson_nlu:
            return self.extract_entities_fallback(text)
        
        try:
            response = self.watson_nlu.analyze(
                text=text[:5000],  # Limit text size for API
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
                
                # Only include high-confidence entities
                if confidence > 0.5:
                    if entity_type in ['person', 'organization', 'company']:
                        entities['parties'].append(entity_text)
                    elif entity_type in ['date', 'datetime', 'time']:
                        entities['dates'].append(entity_text)
                    elif entity_type in ['money', 'currency', 'quantity']:
                        entities['monetary_values'].append(entity_text)
            
            # Extract keywords as legal terms
            keywords = response.get('keywords', [])
            legal_keywords = []
            for keyword in keywords:
                kw_text = keyword.get('text', '')
                kw_relevance = keyword.get('relevance', 0)
                if kw_relevance > 0.5:  # High relevance keywords only
                    legal_keywords.append(kw_text)
            
            entities['legal_terms'] = legal_keywords
            
            # Extract obligations using sentence analysis
            try:
                sentences = sent_tokenize(text)
                obligation_words = ['shall', 'must', 'agrees', 'obligated', 'required', 'responsible']
                for sentence in sentences:
                    if any(word in sentence.lower() for word in obligation_words) and len(sentence) < 200:
                        entities['obligations'].append(sentence.strip())
            except Exception:
                pass
            
            # Remove duplicates and limit results
            for key in entities:
                if isinstance(entities[key], list):
                    entities[key] = list(set(entities[key]))[:5]
            
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
            # Extract dates - simplified pattern
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ]
            
            for pattern in date_patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    entities['dates'].extend(matches)
                except re.error:
                    continue
            
            # Extract monetary values - simplified pattern
            money_patterns = [
                r'\$[\d,]+(?:\.\d{2})?',  # $1,000.00
                r'USD\s*[\d,]+',  # USD 1000
                r'[\d,]+\s*dollars?'  # 1000 dollars
            ]
            
            for pattern in money_patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    entities['monetary_values'].extend(matches)
                except re.error:
                    continue
            
            # Extract common legal terms
            legal_terms = ['agreement', 'contract', 'party', 'parties', 'obligation', 'liability', 
                          'termination', 'breach', 'confidential', 'proprietary', 'indemnification']
            for term in legal_terms:
                if term.lower() in text.lower():
                    entities['legal_terms'].append(term)
            
            # Extract potential party names (capitalized phrases) - simplified
            try:
                party_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|LLC|Corp|Company|Ltd))?\b'
                potential_parties = re.findall(party_pattern, text)
                entities['parties'] = list(set([p for p in potential_parties if len(p.split()) <= 4]))[:5]
            except re.error:
                pass
            
            # Extract obligations (sentences with "shall", "must", "agrees")
            try:
                sentences = sent_tokenize(text)
                obligation_words = ['shall', 'must', 'agrees', 'obligated', 'required']
                for sentence in sentences:
                    if any(word in sentence.lower() for word in obligation_words):
                        if len(sentence) < 200:  # Keep it reasonable
                            entities['obligations'].append(sentence.strip())
            except Exception:
                pass
            
            # Remove duplicates and limit results
            for key in entities:
                if isinstance(entities[key], list):
                    entities[key] = list(set(entities[key]))[:5]
            
            return entities
            
        except Exception as e:
            # Return empty structure if everything fails
            return {
                'parties': ['Could not extract'],
                'dates': ['Could not extract'],
                'monetary_values': ['Could not extract'],
                'obligations': ['Could not extract'],
                'legal_terms': ['Could not extract']
            }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Main entity extraction method - tries Watson first, falls back to regex"""
        if self.watson_nlu:
            return self.extract_entities_with_watson(text)
        else:
            return self.extract_entities_fallback(text)