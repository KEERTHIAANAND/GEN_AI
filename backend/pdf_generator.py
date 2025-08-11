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
        
        self.styles.add(ParagraphStyle(
            name='CustomSmall',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_JUSTIFY,
            spaceAfter=4
        ))
    
    def _clean_text_for_pdf(self, text: str) -> str:
        """Clean text for PDF generation"""
        # Remove or replace problematic characters
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = text.replace('\n\n', '<br/><br/>')
        text = text.replace('\n', '<br/>')
        # Escape XML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('&lt;br/&gt;', '<br/>')
        return text
    
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
        
        story = []
        
        # Title
        title = Paragraph("ClauseWise AI Legal Document Analysis Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Metadata table
        metadata = [
            ["Analysis Date:", datetime.now().strftime("%B %d, %Y at %I:%M %p")],
            ["Document Type:", document_type],
            ["Document Length:", f"{len(original_text.split())} words"],
            ["Clauses Analyzed:", str(len(clauses))]
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        clean_summary = self._clean_text_for_pdf(summary)
        summary_para = Paragraph(clean_summary, self.styles['CustomBody'])
        story.append(summary_para)
        story.append(Spacer(1, 15))
        
        # Key Information Extracted
        story.append(Paragraph("Key Information Extracted", self.styles['CustomHeading']))
        
        info_data = []
        for entity_type, entity_list in entities.items():
            if entity_list and entity_list[0] not in ['Analysis completed', 'See document', 'Could not extract']:
                entity_title = entity_type.replace('_', ' ').title()
                entity_text = ', '.join(entity_list[:3])
                info_data.append([entity_title + ":", entity_text])
        
        if info_data:
            info_table = Table(info_data, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            story.append(info_table)
        else:
            story.append(Paragraph("Key information extracted and available in full analysis.", self.styles['CustomBody']))
        
        story.append(Spacer(1, 15))
        
        # Clause Analysis
        story.append(Paragraph("AI-Powered Clause Analysis", self.styles['CustomHeading']))
        
        for i, (clause, explanation) in enumerate(zip(clauses[:5], clause_explanations[:5])):
            story.append(Paragraph(f"Clause {i+1}:", self.styles['CustomHeading']))
            
            # Original clause (truncated)
            clause_text = clause[:400] + "..." if len(clause) > 400 else clause
            clean_clause = self._clean_text_for_pdf(clause_text)
            story.append(Paragraph(f"<b>Original Text:</b><br/>{clean_clause}", self.styles['CustomSmall']))
            
            # Explanation
            clean_explanation = self._clean_text_for_pdf(explanation)
            story.append(Paragraph(f"<b>Plain English Explanation:</b><br/>{clean_explanation}", self.styles['CustomBody']))
            story.append(Spacer(1, 10))
        
        # Simplified Document Preview
        story.append(Paragraph("Simplified Document (Preview)", self.styles['CustomHeading']))
        simplified_preview = simplified_text[:1500] + "..." if len(simplified_text) > 1500 else simplified_text
        clean_simplified = self._clean_text_for_pdf(simplified_preview)
        story.append(Paragraph(clean_simplified, self.styles['CustomSmall']))
        story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("Important Legal Disclaimer", self.styles['CustomHeading']))
        disclaimer_text = """
        This AI-generated analysis is provided for informational purposes only and does not constitute legal advice. 
        The analysis may contain errors or omissions. Always consult with a qualified attorney before making any legal decisions. 
        The accuracy and completeness of this analysis is not guaranteed.
        """
        story.append(Paragraph(disclaimer_text, self.styles['CustomBody']))
        
        # Footer
        footer_text = f"Generated by ClauseWise AI Legal Analyzer - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Spacer(1, 10))
        story.append(Paragraph(footer_text, self.styles['CustomSmall']))
        
        # Build PDF
        try:
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            print(f"PDF generation error: {e}")
            # Return a simple error PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            error_story = [
                Paragraph("ClauseWise Analysis Report", self.styles['CustomTitle']),
                Spacer(1, 20),
                Paragraph("Error generating detailed report. Please try again.", self.styles['CustomBody'])
            ]
            doc.build(error_story)
            buffer.seek(0)
            return buffer.getvalue()