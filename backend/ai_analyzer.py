from typing import Dict, List
import re
import json
from config import Config

# IBM Watson imports
try:
    from ibm_watson import NaturalLanguageUnderstandingV1, AssistantV2
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, ConceptsOptions, EmotionOptions, SentimentOptions
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False

# IBM watsonx imports
try:
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False

class AIAnalyzer:
    def __init__(self):
        self.watson_nlu = None
        self.watson_assistant = None
        self.granite_model = None
        
        # Initialize Watson NLU
        if WATSON_AVAILABLE:
            try:
                authenticator = IAMAuthenticator(Config.WATSON_NLU_API_KEY)
                self.watson_nlu = NaturalLanguageUnderstandingV1(
                    version='2022-04-07',
                    authenticator=authenticator
                )
                self.watson_nlu.set_service_url(Config.WATSON_NLU_URL)
            except Exception as e:
                print(f"Watson NLU initialization failed: {e}")
        
        # Initialize Granite Model
        if WATSONX_AVAILABLE:
            try:
                self.granite_model = Model(
                    model_id=Config.GRANITE_MODEL_ID,
                    params={
                        GenParams.DECODING_METHOD: "greedy",
                        GenParams.MAX_NEW_TOKENS: 500,
                        GenParams.MIN_NEW_TOKENS: 1,
                        GenParams.TEMPERATURE: 0.1,
                        GenParams.TOP_K: 50,
                        GenParams.TOP_P: 1
                    },
                    credentials={
                        "apikey": Config.WATSONX_API_KEY,
                        "url": Config.WATSONX_URL
                    },
                    project_id=Config.WATSONX_PROJECT_ID
                )
            except Exception as e:
                print(f"Granite model initialization failed: {e}")
        
        # Fallback document keywords for classification
        self.document_keywords = {
            'NDA (Non-Disclosure Agreement)': [
                'confidential', 'proprietary', 'non-disclosure', 'confidentiality',
                'trade secret', 'proprietary information', 'confidential information'
            ],
            'Employment Contract': [
                'employment', 'employee', 'employer', 'salary', 'wages', 'termination',
                'job duties', 'work schedule', 'benefits', 'vacation'
            ],
            'Service Agreement': [
                'services', 'service provider', 'client', 'deliverables', 'scope of work',
                'service fees', 'performance', 'service agreement'
            ],
            'Lease Agreement': [
                'lease', 'tenant', 'landlord', 'rent', 'premises', 'property',
                'rental', 'lease term', 'security deposit'
            ],
            'Purchase Agreement': [
                'purchase', 'buyer', 'seller', 'sale', 'goods', 'merchandise',
                'purchase price', 'delivery', 'title transfer'
            ],
            'Partnership Agreement': [
                'partnership', 'partners', 'profit', 'loss', 'capital contribution',
                'business partnership', 'joint venture'
            ],
            'License Agreement': [
                'license', 'licensor', 'licensee', 'intellectual property',
                'copyright', 'trademark', 'patent', 'licensing'
            ]
        }
    
    def classify_document_with_watson(self, text: str) -> str:
        """Classify document using Watson NLU"""
        if not self.watson_nlu:
            return self.classify_document_fallback(text)
        
        try:
            response = self.watson_nlu.analyze(
                text=text[:3000],  # Limit text size for API
                features=Features(
                    keywords=KeywordsOptions(limit=10),
                    concepts=ConceptsOptions(limit=5)
                )
            ).get_result()
            
            # Extract keywords and concepts for classification
            keywords = [kw['text'].lower() for kw in response.get('keywords', [])]
            concepts = [c['text'].lower() for c in response.get('concepts', [])]
            
            all_terms = keywords + concepts
            
            # Score against document types
            scores = {}
            for doc_type, type_keywords in self.document_keywords.items():
                score = sum(1 for keyword in type_keywords if any(term in keyword or keyword in term for term in all_terms))
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
            score = sum(1 for keyword in keywords if keyword in text_lower)
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
    
    def simplify_text_with_granite(self, text: str) -> str:
        """Simplify text using Granite model"""
        if not self.granite_model:
            return self.simplify_text_fallback(text)
        
        try:
            # Create prompt for text simplification
            prompt = f"""
            Please simplify the following legal text into plain English that anyone can understand. 
            Keep the meaning intact but make it more accessible:

            {text[:2000]}

            Simplified version:
            """
            
            response = self.granite_model.generate_text(prompt=prompt)
            if response:
                # Extract the simplified text from response
                simplified = response.strip()
                # Remove the prompt echo if present
                if "Simplified version:" in simplified:
                    simplified = simplified.split("Simplified version:")[-1].strip()
                return simplified if simplified else self.simplify_text_fallback(text)
            else:
                return self.simplify_text_fallback(text)
                
        except Exception as e:
            print(f"Granite simplification failed: {e}")
            return self.simplify_text_fallback(text)
    
    def simplify_text_fallback(self, text: str) -> str:
        """Fallback simplification using rule-based approach"""
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
            r'\bforce majeure\b': 'uncontrollable circumstances',
            r'\bquid pro quo\b': 'something for something',
            r'\bprima facie\b': 'at first look'
        }
        
        simplified = text
        for legal_term, simple_term in simplifications.items():
            simplified = re.sub(legal_term, simple_term, simplified, flags=re.IGNORECASE)
        
        return simplified
    
    def simplify_text(self, text: str) -> str:
        """Main text simplification method"""
        if self.granite_model:
            return self.simplify_text_with_granite(text)
        else:
            return self.simplify_text_fallback(text)
    
    def explain_clause_with_granite(self, clause: str) -> str:
        """Generate clause explanation using Granite model"""
        if not self.granite_model:
            return self.explain_clause_fallback(clause)
        
        try:
            prompt = f"""
            Explain this legal clause in simple, everyday language that anyone can understand:

            "{clause[:500]}"

            Explanation in plain English:
            """
            
            response = self.granite_model.generate_text(prompt=prompt)
            if response:
                explanation = response.strip()
                if "Explanation in plain English:" in explanation:
                    explanation = explanation.split("Explanation in plain English:")[-1].strip()
                return explanation if explanation else self.explain_clause_fallback(clause)
            else:
                return self.explain_clause_fallback(clause)
                
        except Exception as e:
            print(f"Granite clause explanation failed: {e}")
            return self.explain_clause_fallback(clause)
    
    def explain_clause_fallback(self, clause: str) -> str:
        """Fallback clause explanation"""
        clause_lower = clause.lower()
        
        if any(word in clause_lower for word in ['confidential', 'non-disclosure', 'proprietary']):
            return "This clause requires keeping certain information secret and not sharing it with others."
        elif any(word in clause_lower for word in ['termination', 'terminate', 'end']):
            return "This clause explains how and when the agreement can be ended."
        elif any(word in clause_lower for word in ['liability', 'liable', 'responsible']):
            return "This clause defines who is responsible for damages or problems that may occur."
        elif any(word in clause_lower for word in ['payment', 'fee', 'cost', 'money']):
            return "This clause covers payment terms, amounts, and when payments are due."
        elif any(word in clause_lower for word in ['breach', 'violation', 'default']):
            return "This clause explains what happens if someone doesn't follow the agreement."
        elif any(word in clause_lower for word in ['force majeure', 'uncontrollable', 'acts of god']):
            return "This clause covers situations beyond anyone's control that prevent fulfilling the agreement."
        elif any(word in clause_lower for word in ['indemnify', 'hold harmless']):
            return "This clause means one party will protect the other from legal claims or losses."
        elif any(word in clause_lower for word in ['governing law', 'jurisdiction']):
            return "This clause determines which state's or country's laws apply to the agreement."
        else:
            if len(clause) > 200:
                return "This is a detailed clause that requires careful review with legal counsel."
            else:
                return "This clause sets out specific terms and conditions for the agreement."
    
    def explain_clause(self, clause: str) -> str:
        """Main clause explanation method"""
        if self.granite_model:
            return self.explain_clause_with_granite(clause)
        else:
            return self.explain_clause_fallback(clause)
    
    def extract_entities_with_watson(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using Watson NLU"""
        if not self.watson_nlu:
            return self.extract_entities_fallback(text)
        
        try:
            response = self.watson_nlu.analyze(
                text=text[:5000],  # Limit text size
                features=Features(
                    entities=EntitiesOptions(limit=20),
                    keywords=KeywordsOptions(limit=15)
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
                entity_text = entity.get('text', '')
                
                if entity_type in ['person', 'organization', 'company']:
                    entities['parties'].append(entity_text)
                elif entity_type in ['date', 'datetime']:
                    entities['dates'].append(entity_text)
                elif entity_type in ['money', 'currency']:
                    entities['monetary_values'].append(entity_text)
            
            # Extract keywords as legal terms
            keywords = response.get('keywords', [])
            for keyword in keywords:
                entities['legal_terms'].append(keyword.get('text', ''))
            
            # Limit results
            for key in entities:
                entities[key] = entities[key][:5]
            
            return entities
            
        except Exception as e:
            print(f"Watson entity extraction failed: {e}")
            return self.extract_entities_fallback(text)
    
    def extract_entities_fallback(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity extraction"""
        entities = {
            'parties': ['Could not extract with current setup'],
            'dates': ['Could not extract with current setup'],
            'monetary_values': ['Could not extract with current setup'],
            'obligations': ['Could not extract with current setup'],
            'legal_terms': ['Could not extract with current setup']
        }
        return entities
    
    def generate_summary(self, text: str, entities: Dict, document_type: str) -> str:
        """Generate document summary"""
        summary = f"**Document Type:** {document_type}\n\n"
        
        if entities.get('parties') and entities['parties'][0] != 'Could not extract with current setup':
            summary += f"**Key Parties:** {', '.join(entities['parties'][:3])}\n\n"
        
        if entities.get('dates') and entities['dates'][0] != 'Could not extract with current setup':
            summary += f"**Important Dates:** {', '.join(entities['dates'][:3])}\n\n"
        
        if entities.get('monetary_values') and entities['monetary_values'][0] != 'Could not extract with current setup':
            summary += f"**Financial Terms:** {', '.join(entities['monetary_values'][:3])}\n\n"
        
        # Word count and complexity
        word_count = len(text.split())
        summary += f"**Document Length:** {word_count} words\n\n"
        
        # AI Enhancement Status
        if self.watson_nlu and self.granite_model:
            summary += "**AI Enhancement:** ✅ IBM Watson NLU + Granite Model\n\n"
        elif self.watson_nlu:
            summary += "**AI Enhancement:** ✅ IBM Watson NLU\n\n"
        elif self.granite_model:
            summary += "**AI Enhancement:** ✅ IBM Granite Model\n\n"
        else:
            summary += "**AI Enhancement:** ⚠️ Using fallback analysis\n\n"
        
        summary += "**Key Legal Areas:** "
        if 'confidential' in text.lower():
            summary += "Confidentiality, "
        if any(word in text.lower() for word in ['liability', 'indemnify']):
            summary += "Liability, "
        if any(word in text.lower() for word in ['termination', 'breach']):
            summary += "Termination, "
        if any(word in text.lower() for word in ['payment', 'fee']):
            summary += "Financial Terms, "
        
        summary = summary.rstrip(', ') + "\n"
        
        return summary