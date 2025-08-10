from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime
import io
from typing import Dict, List

class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=12,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))
    
    def generate_analysis_report(self, 
                               original_text: str,
                               simplified_text: str,
                               clauses: List[str],
                               clause_explanations: List[str],
                               entities: Dict[str, List[str]],
                               document_type: str,
                               summary: str) -> bytes:
        """Generate PDF report of document analysis"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=inch, bottomMargin=inch)
        
        # Build content
        story = []
        
        # Title
        title = Paragraph("ClauseWise Legal Document Analysis Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Metadata
        metadata = [
            ["Analysis Date:", datetime.now().strftime("%B %d, %Y at %I:%M %p")],
            ["Document Type:", document_type],
            ["Document Length:", f"{len(original_text.split())} words"]
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Document Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        summary_para = Paragraph(summary.replace('**', '').replace('\\n', '<br/>'), self.styles['CustomBody'])
        story.append(summary_para)
        story.append(Spacer(1, 15))
        
        # Named Entities
        story.append(Paragraph("Extracted Information", self.styles['CustomHeading']))
        
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_title = entity_type.replace('_', ' ').title()
                story.append(Paragraph(f"<b>{entity_title}:</b>", self.styles['CustomBody']))
                for entity in entity_list[:5]:  # Limit to first 5
                    story.append(Paragraph(f"• {entity}", self.styles['CustomBody']))
                story.append(Spacer(1, 10))
        
        # Key Clauses Analysis
        story.append(Paragraph("Key Clauses Analysis", self.styles['CustomHeading']))
        
        for i, (clause, explanation) in enumerate(zip(clauses[:5], clause_explanations[:5])):
            story.append(Paragraph(f"<b>Clause {i+1}:</b>", self.styles['CustomBody']))
            
            # Original clause (truncated)
            clause_text = clause[:300] + "..." if len(clause) > 300 else clause
            story.append(Paragraph(f"<i>Original:</i> {clause_text}", self.styles['CustomBody']))
            
            # Explanation
            story.append(Paragraph(f"<i>Plain English:</i> {explanation}", self.styles['CustomBody']))
            story.append(Spacer(1, 10))
        
        # Simplified Document (first 1000 characters)
        story.append(Paragraph("Document Summary (Simplified)", self.styles['CustomHeading']))
        simplified_preview = simplified_text[:1000] + "..." if len(simplified_text) > 1000 else simplified_text
        story.append(Paragraph(simplified_preview, self.styles['CustomBody']))
        story.append(Spacer(1, 15))
        
        # Disclaimer
        story.append(Paragraph("Important Disclaimer", self.styles['CustomHeading']))
        disclaimer_text = '''
        This analysis is generated by AI and is for informational purposes only. 
        It does not constitute legal advice. Always consult with a qualified attorney 
        for legal matters. The accuracy of this analysis is not guaranteed, and 
        important details may be missed or misinterpreted.
        '''
        story.append(Paragraph(disclaimer_text, self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()