from typing import Dict, List
import re
import requests
import json
from config import Config

# IBM Watson imports
try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, ConceptsOptions
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False
    print("Watson SDK not available")

class AIAnalyzer:
    def __init__(self):
        self.watson_nlu = None
        self.huggingface_api_key = Config.HUGGINGFACE_API_KEY
        self.huggingface_api_url = "https://api-inference.huggingface.co/models/"
        
        # Initialize Watson NLU
        if WATSON_AVAILABLE and Config.WATSON_NLU_API_KEY:
            try:
                authenticator = IAMAuthenticator(Config.WATSON_NLU_API_KEY)
                self.watson_nlu = NaturalLanguageUnderstandingV1(
                    version='2022-04-07',
                    authenticator=authenticator
                )
                self.watson_nlu.set_service_url(Config.WATSON_NLU_URL)
                print("Watson NLU initialized successfully in AIAnalyzer")
            except Exception as e:
                print(f"Watson NLU initialization failed: {e}")
        
        # Document classification keywords
        self.document_keywords = {
            'NDA (Non-Disclosure Agreement)': [
                'confidential', 'proprietary', 'non-disclosure', 'confidentiality',
                'trade secret', 'confidential information'
            ],
            'Employment Contract': [
                'employment', 'employee', 'employer', 'salary', 'wages', 'termination',
                'job duties', 'benefits'
            ],
            'Service Agreement': [
                'services', 'service provider', 'client', 'deliverables', 'scope of work'
            ],
            'Lease Agreement': [
                'lease', 'tenant', 'landlord', 'rent', 'premises', 'property'
            ],
            'Purchase Agreement': [
                'purchase', 'buyer', 'seller', 'sale', 'goods', 'purchase price'
            ],
            'Partnership Agreement': [
                'partnership', 'partners', 'profit', 'loss', 'capital contribution'
            ],
            'License Agreement': [
                'license', 'licensor', 'licensee', 'intellectual property'
            ]
        }
    
    def query_huggingface_api(self, model_name: str, payload: dict, max_retries: int = 3) -> dict:
        """Query Hugging Face API with retry mechanism"""
        headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.huggingface_api_url}{model_name}",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    print(f"Model loading, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(10)
                        continue
                else:
                    print(f"HuggingFace API error: {response.status_code} - {response.text}")
                    break
                    
            except Exception as e:
                print(f"HuggingFace API request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)
        
        return None
    
    def classify_document_with_watson(self, text: str) -> str:
        """Classify document using Watson NLU"""
        if not self.watson_nlu:
            return self.classify_document_fallback(text)
        
        try:
            analysis_text = text[:3000] if len(text) > 3000 else text
            
            response = self.watson_nlu.analyze(
                text=analysis_text,
                features=Features(
                    keywords=KeywordsOptions(limit=10),
                    concepts=ConceptsOptions(limit=5)
                )
            ).get_result()
            
            keywords = [kw['text'].lower() for kw in response.get('keywords', [])]
            concepts = [c['text'].lower() for c in response.get('concepts', [])]
            
            all_terms = keywords + concepts + text.lower().split()
            
            # Score against document types
            scores = {}
            for doc_type, type_keywords in self.document_keywords.items():
                score = sum(1 for keyword in type_keywords if any(keyword in term or term in keyword for term in all_terms))
                scores[doc_type] = score
            
            if max(scores.values()) == 0:
                return 'Other Legal Document'
            
            return max(scores, key=scores.get)
            
        except Exception as e:
            print(f"Watson classification failed: {e}")
            return self.classify_document_fallback(text)
    
    def classify_document_fallback(self, text: str) -> str:
        """Fallback classification method"""
        text_lower = text.lower()
        scores = {}
        
        for doc_type, keywords in self.document_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            scores[doc_type] = score
        
        if max(scores.values()) == 0:
            return 'Other Legal Document'
        
        return max(scores, key=scores.get)
    
    def classify_document(self, text: str) -> str:
        """Main classification method"""
        if self.watson_nlu:
            return self.classify_document_with_watson(text)
        else:
            return self.classify_document_fallback(text)
    
    def simplify_text_with_huggingface(self, text: str) -> str:
        """Simplify text using Hugging Face API"""
        if not self.huggingface_api_key:
            return self.simplify_text_fallback(text)
        
        try:
            # Split text into chunks if too long
            text_chunks = []
            if len(text) > 1000:
                # Split into paragraphs or sentences
                chunks = text.split('\n\n')
                current_chunk = ""
                for chunk in chunks:
                    if len(current_chunk + chunk) < 1000:
                        current_chunk += chunk + "\n\n"
                    else:
                        if current_chunk:
                            text_chunks.append(current_chunk.strip())
                        current_chunk = chunk + "\n\n"
                if current_chunk:
                    text_chunks.append(current_chunk.strip())
            else:
                text_chunks = [text]
            
            simplified_parts = []
            
            # Try summarization model first
            for chunk in text_chunks[:3]:  # Limit to 3 chunks
                try:
                    payload = {
                        "inputs": f"Simplify this legal text into plain English: {chunk[:500]}",
                        "parameters": {
                            "max_length": 300,
                            "min_length": 50,
                            "do_sample": False
                        }
                    }
                    
                    result = self.query_huggingface_api(Config.HUGGINGFACE_SUMMARIZATION_MODEL, payload)
                    
                    if result and isinstance(result, list) and len(result) > 0:
                        summary_text = result[0].get('summary_text', '')
                        if summary_text and len(summary_text) > 20:
                            simplified_parts.append(summary_text)
                        else:
                            simplified_parts.append(self.simplify_text_fallback(chunk))
                    else:
                        simplified_parts.append(self.simplify_text_fallback(chunk))
                        
                except Exception as e:
                    print(f"Error with HuggingFace chunk processing: {e}")
                    simplified_parts.append(self.simplify_text_fallback(chunk))
            
            if simplified_parts:
                result = "\n\n".join(simplified_parts)
                intro = "**AI-Simplified Legal Document**\n\nThis document has been simplified using advanced AI for easier understanding:\n\n"
                return intro + result if len(result) > 50 else self.simplify_text_fallback(text)
            else:
                return self.simplify_text_fallback(text)
                
        except Exception as e:
            print(f"HuggingFace simplification failed: {e}")
            return self.simplify_text_fallback(text)
    
    def simplify_text_fallback(self, text: str) -> str:
        """Enhanced fallback simplification"""
        simplifications = {
            r'\bheretofore\b': 'before this',
            r'\bhereinafter\b': 'from now on',
            r'\bwhereas\b': 'since',
            r'\btherefore\b': 'so',
            r'\bnotwithstanding\b': 'despite',
            r'\bforthwith\b': 'immediately',
            r'\bpursuant to\b': 'according to',
            r'\bin consideration of\b': 'in exchange for',
            r'\bshall be deemed\b': 'will be considered',
            r'\bshall have the right\b': 'may',
            r'\bshall be entitled\b': 'has the right',
            r'\bshall not be liable\b': 'is not responsible',
            r'\bforce majeure\b': 'unexpected events beyond control',
            r'\bindemnify\b': 'protect from legal claims',
            r'\bhold harmless\b': 'protect from responsibility',
            r'\bwhereas\b': 'given that',
            r'\bthereby\b': 'by doing this',
            r'\bwherein\b': 'in which',
            r'\bwhereby\b': 'by which'
        }
        
        simplified = text
        for legal_term, simple_term in simplifications.items():
            simplified = re.sub(legal_term, simple_term, simplified, flags=re.IGNORECASE)
        
        # Add sections for better readability
        intro = "**Simplified Legal Document**\n\nThis document has been simplified for easier understanding:\n\n"
        
        # Break into paragraphs if very long
        if len(simplified) > 1000:
            paragraphs = simplified.split('. ')
            formatted_paragraphs = []
            current_para = ""
            
            for sentence in paragraphs:
                current_para += sentence + ". "
                if len(current_para) > 200:
                    formatted_paragraphs.append(current_para.strip())
                    current_para = ""
            
            if current_para:
                formatted_paragraphs.append(current_para.strip())
            
            simplified = "\n\n".join(formatted_paragraphs)
        
        return intro + simplified
    
    def simplify_text(self, text: str) -> str:
        """Main text simplification method"""
        if self.huggingface_api_key:
            return self.simplify_text_with_huggingface(text)
        else:
            return self.simplify_text_fallback(text)
    
    def explain_clause_with_huggingface(self, clause: str) -> str:
        """Generate clause explanation using Hugging Face"""
        if not self.huggingface_api_key:
            return self.explain_clause_fallback(clause)
        
        try:
            # Limit clause length
            clause_text = clause[:400] if len(clause) > 400 else clause
            
            # Try to get explanation using text generation
            payload = {
                "inputs": f"Explain this legal clause in simple terms: {clause_text}",
                "parameters": {
                    "max_length": 200,
                    "min_length": 30,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            result = self.query_huggingface_api("microsoft/DialoGPT-medium", payload)
            
            if result and isinstance(result, list) and len(result) > 0:
                explanation = result[0].get('generated_text', '')
                if explanation and len(explanation) > 20:
                    # Clean up the explanation
                    explanation = explanation.replace(f"Explain this legal clause in simple terms: {clause_text}", "").strip()
                    return explanation if explanation else self.explain_clause_fallback(clause)
                else:
                    return self.explain_clause_fallback(clause)
            else:
                return self.explain_clause_fallback(clause)
                
        except Exception as e:
            print(f"HuggingFace clause explanation failed: {e}")
            return self.explain_clause_fallback(clause)
    
    def explain_clause_fallback(self, clause: str) -> str:
        """Enhanced fallback clause explanation"""
        clause_lower = clause.lower()
        
        explanations = {
            ['confidential', 'non-disclosure', 'proprietary']: "This section requires keeping certain information secret and not sharing it with others.",
            ['termination', 'terminate', 'end']: "This explains how and when the agreement can be ended by either party.",
            ['liability', 'liable', 'responsible', 'damages']: "This defines who is responsible for problems or damages that may occur.",
            ['payment', 'fee', 'cost', 'money', 'compensation']: "This covers payment terms, amounts, and when payments are due.",
            ['breach', 'violation', 'default']: "This explains what happens if someone doesn't follow the agreement.",
            ['force majeure', 'uncontrollable', 'acts of god']: "This covers situations beyond anyone's control that prevent fulfilling the agreement.",
            ['indemnify', 'hold harmless']: "This means one party will protect the other from legal claims or financial losses.",
            ['governing law', 'jurisdiction']: "This determines which state's or country's laws apply to this agreement.",
            ['intellectual property', 'copyright', 'patent']: "This covers ownership and use of ideas, inventions, or creative works.",
            ['warranty', 'guarantee']: "This is a promise about the quality or performance of something.",
            ['assignment', 'transfer']: "This covers whether and how the agreement can be transferred to someone else."
        }
        
        for keywords, explanation in explanations.items():
            if any(word in clause_lower for word in keywords):
                return explanation
        
        # Default explanation based on clause length
        if len(clause) > 200:
            return "This is a detailed legal provision that sets out specific terms and conditions. Consider reviewing with legal counsel for full understanding."
        else:
            return "This clause establishes specific terms and obligations under this agreement."
    
    def explain_clause(self, clause: str) -> str:
        """Main clause explanation method"""
        if self.huggingface_api_key:
            return self.explain_clause_with_huggingface(clause)
        else:
            return self.explain_clause_fallback(clause)
    
    def generate_summary(self, text: str, entities: Dict, document_type: str) -> str:
        """Generate comprehensive document summary"""
        summary = f"**Document Type:** {document_type}\n\n"
        
        # Add party information if available
        if entities.get('parties') and entities['parties'][0] not in ['Analysis completed', 'Document analysis completed']:
            parties_text = ', '.join(entities['parties'][:3])
            summary += f"**Key Parties:** {parties_text}\n\n"
        
        # Add date information if available
        if entities.get('dates') and entities['dates'][0] not in ['See document', 'See full document']:
            dates_text = ', '.join(entities['dates'][:2])
            summary += f"**Important Dates:** {dates_text}\n\n"
        
        # Add financial information if available
        if entities.get('monetary_values') and entities['monetary_values'][0] not in ['See document', 'See full document']:
            money_text = ', '.join(entities['monetary_values'][:2])
            summary += f"**Financial Terms:** {money_text}\n\n"
        
        # Document metrics
        word_count = len(text.split())
        summary += f"**Document Length:** {word_count} words\n\n"
        
        # AI Enhancement Status
        if self.watson_nlu and self.huggingface_api_key:
            summary += "**AI Analysis:** ✅ Premium (Watson NLU + Hugging Face)\n\n"
        elif self.watson_nlu:
            summary += "**AI Analysis:** ✅ Enhanced (Watson NLU)\n\n"
        elif self.huggingface_api_key:
            summary += "**AI Analysis:** ✅ Advanced (Hugging Face)\n\n"
        else:
            summary += "**AI Analysis:** ⚙️ Standard (Rule-based)\n\n"
        
        # Key legal areas
        legal_areas = []
        text_lower = text.lower()
        if any(word in text_lower for word in ['confidential', 'proprietary']):
            legal_areas.append("Confidentiality")
        if any(word in text_lower for word in ['liability', 'indemnify']):
            legal_areas.append("Liability")
        if any(word in text_lower for word in ['termination', 'breach']):
            legal_areas.append("Termination")
        if any(word in text_lower for word in ['payment', 'fee', 'compensation']):
            legal_areas.append("Financial Terms")
        if any(word in text_lower for word in ['intellectual property', 'copyright', 'patent']):
            legal_areas.append("Intellectual Property")
        
        if legal_areas:
            summary += f"**Key Legal Areas:** {', '.join(legal_areas)}"
        
        return summary