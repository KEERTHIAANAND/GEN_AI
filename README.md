# ClauseWise Legal Document Analyzer

## Overview
ClauseWise is an AI-powered legal document analyzer that simplifies, decodes, and classifies complex legal texts. Built for lawyers, businesses, and laypersons to better understand legal contracts and documents.

## Features
- **Clause Simplification**: Converts complex legal language into plain English
- **Named Entity Recognition**: Extracts parties, dates, monetary values, and legal terms
- **Document Classification**: Identifies document types (NDA, employment contracts, etc.)
- **Clause Breakdown**: Segments and analyzes individual clauses
- **Multi-format Support**: PDF, DOCX, and TXT files
- **PDF Report Generation**: Downloadable analysis reports
- **User-friendly Interface**: Built with Streamlit

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Extract the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data (automatically handled on first run):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using ClauseWise
1. **Accept Terms and Conditions**
2. **Upload a legal document** (PDF, DOCX, or TXT)
3. **Click "Analyze Document"**
4. **Review results** in the tabbed interface:
   - Summary: Overview and key metrics
   - Key Information: Extracted entities
   - Clause Analysis: Detailed clause breakdown
   - Simplified Text: Plain English version
   - Download Report: Generate PDF report

## Project Structure
```
clausewise/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── backend/
    ├── __init__.py
    ├── document_processor.py   # Text extraction and processing
    ├── ai_analyzer.py         # AI analysis and classification
    └── pdf_generator.py       # PDF report generation
```

## Technical Architecture

### Backend Components
- **DocumentProcessor**: Handles text extraction from various file formats
- **AIAnalyzer**: Performs document classification and text simplification
- **PDFGenerator**: Creates downloadable analysis reports

### AI Analysis Features
- Document type classification using keyword matching
- Legal term simplification using pattern replacement
- Clause extraction using regex patterns
- Named entity recognition for legal documents

## Supported Document Types
- NDA (Non-Disclosure Agreement)
- Employment Contract
- Service Agreement
- Lease Agreement
- Purchase Agreement
- Partnership Agreement
- License Agreement
- Other Legal Documents

## File Support
- **PDF**: Text extraction using PyPDF2
- **DOCX**: Text extraction using python-docx
- **TXT**: Direct text processing
- **Size Limit**: 10MB per file

## Dependencies
- streamlit: Web application framework
- PyPDF2: PDF text extraction
- python-docx: DOCX text extraction
- nltk: Natural language processing
- reportlab: PDF generation
- pandas, numpy: Data processing

## Limitations
- AI analysis is for informational purposes only
- Not a substitute for professional legal advice
- Text extraction quality depends on document format
- Complex legal concepts may not be fully captured

## Development Notes

### For Hackathon Participants
This is a complete, working implementation ready for demonstration. Key features:
- Responsive web interface
- Real-time document processing
- Professional PDF report generation
- Comprehensive error handling
- Terms and conditions compliance

### Customization
- Modify `Config` class for different settings
- Update document type keywords in `AIAnalyzer`
- Customize PDF report layout in `PDFGenerator`
- Add new simplification rules in `AIAnalyzer.simplify_text()`

## Legal Disclaimer
This tool is for educational and informational purposes only. The analysis provided does not constitute legal advice. Always consult with qualified legal counsel for legal matters.

## License
Educational/Hackathon use only.

## Support
For technical issues or questions, refer to the documentation or check the error messages in the Streamlit interface.