import streamlit as st
import io
from datetime import datetime
from config import Config
from backend.document_processor import DocumentProcessor
from backend.ai_analyzer import AIAnalyzer
from backend.pdf_generator import PDFGenerator

# Page config
st.set_page_config(
    page_title="ClauseWise Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return (
        DocumentProcessor(),
        AIAnalyzer(),
        PDFGenerator()
    )

def display_api_status(ai_analyzer, doc_processor):
    """Display API connection status"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔌 AI Services Status")
        
        # Watson NLU Status
        if ai_analyzer.watson_nlu and doc_processor.watson_nlu:
            st.success("✅ Watson NLU Connected")
        else:
            st.warning("⚠️ Watson NLU: Using Fallback")
        
        # Granite Model Status
        if ai_analyzer.granite_model:
            st.success("✅ Granite Model Connected")
        else:
            st.warning("⚠️ Granite: Using Fallback")

def main():
    doc_processor, ai_analyzer, pdf_generator = init_components()
    
    # Header
    st.title("⚖️ ClauseWise Legal Document Analyzer")
    st.markdown("*Advanced AI-powered legal document analysis with IBM Watson & Granite*")
    
    # Display API status
    display_api_status(ai_analyzer, doc_processor)
    
    st.divider()
    
    # Terms and Conditions
    if 'terms_accepted' not in st.session_state:
        st.session_state.terms_accepted = False
    
    if not st.session_state.terms_accepted:
        st.header("📋 Terms and Conditions")
        
        with st.expander("Please read and accept the Terms and Conditions", expanded=True):
            st.markdown(Config.TERMS_CONDITIONS)
            
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ I Accept Terms and Conditions", type="primary", use_container_width=True):
                st.session_state.terms_accepted = True
                st.rerun()
        
        st.stop()
    
    # Main application
    # Sidebar
    st.sidebar.header("📄 Document Upload")
    st.sidebar.markdown(f"**Supported formats:** {', '.join(Config.SUPPORTED_FORMATS).upper()}")
    st.sidebar.markdown(f"**Max file size:** {Config.MAX_FILE_SIZE // (1024*1024)}MB")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a legal document",
        type=Config.SUPPORTED_FORMATS,
        help="Upload a PDF, DOCX, or TXT file for AI analysis"
    )
    
    if uploaded_file is not None:
        # File validation
        if uploaded_file.size > Config.MAX_FILE_SIZE:
            st.sidebar.error(f"File size exceeds {Config.MAX_FILE_SIZE // (1024*1024)}MB limit")
            st.stop()
        
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Display file info
        st.sidebar.success(f"📄 File loaded: {uploaded_file.name}")
        st.sidebar.info(f"📊 Size: {uploaded_file.size / 1024:.1f} KB")
        
        # Process button
        if st.sidebar.button("🚀 Analyze with AI", type="primary", use_container_width=True):
            # Analysis progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Analyzing document with advanced AI..."):
                try:
                    # Step 1: Extract text
                    status_text.text("📖 Extracting text from document...")
                    progress_bar.progress(10)
                    
                    raw_text = doc_processor.extract_text(uploaded_file, file_type)
                    
                    if not raw_text or len(raw_text.strip()) < 50:
                        st.error("❌ Document appears to be empty or text extraction failed. Please try a different file.")
                        st.stop()
                    
                    clean_text = doc_processor.clean_text(raw_text)
                    progress_bar.progress(20)
                    
                    if len(clean_text) < 50:
                        st.error("❌ Document text is too short for analysis. Please upload a more substantial document.")
                        st.stop()
                    
                    # Step 2: AI Classification
                    status_text.text("🤖 Classifying document with Watson AI...")
                    progress_bar.progress(40)
                    document_type = ai_analyzer.classify_document(clean_text)
                    
                    # Step 3: Text Simplification
                    status_text.text("📝 Simplifying text with Granite model...")
                    progress_bar.progress(60)
                    simplified_text = ai_analyzer.simplify_text(clean_text)
                    
                    # Step 4: Extract clauses and entities
                    status_text.text("🔍 Extracting key information...")
                    progress_bar.progress(75)
                    
                    try:
                        clauses = doc_processor.extract_clauses(clean_text)
                        if not clauses:
                            clauses = ["Could not extract specific clauses. Please review the full simplified text."]
                    except Exception as e:
                        clauses = [f"Clause extraction encountered an issue: {str(e)}"]
                    
                    try:
                        entities = doc_processor.extract_entities(clean_text)
                    except Exception as e:
                        entities = {
                            'parties': [f'Error: {str(e)[:50]}...'],
                            'dates': ['Could not extract'],
                            'monetary_values': ['Could not extract'],
                            'obligations': ['Could not extract'],
                            'legal_terms': ['Could not extract']
                        }
                    
                    # Step 5: Generate explanations
                    status_text.text("💡 Generating AI explanations...")
                    progress_bar.progress(90)
                    
                    clause_explanations = []
                    for clause in clauses:
                        try:
                            explanation = ai_analyzer.explain_clause(clause)
                            clause_explanations.append(explanation)
                        except Exception as e:
                            clause_explanations.append(f"Could not generate explanation: {str(e)[:50]}...")
                    
                    # Step 6: Generate summary
                    try:
                        summary = ai_analyzer.generate_summary(clean_text, entities, document_type)
                    except Exception as e:
                        summary = f"**Document Type:** {document_type}\n\n**Word Count:** {len(clean_text.split())} words\n\n**Status:** Analysis completed with some limitations: {str(e)[:100]}..."
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Store results in session state
                    st.session_state.analysis_results = {
                        'original_text': clean_text,
                        'simplified_text': simplified_text,
                        'document_type': document_type,
                        'clauses': clauses,
                        'clause_explanations': clause_explanations,
                        'entities': entities,
                        'summary': summary,
                        'filename': uploaded_file.name,
                        'file_size': uploaded_file.size,
                        'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("🎉 Document analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    st.info("💡 Try uploading a different file or check if the document is not corrupted.")
                    st.stop()
    
    # Display results
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        # Results header with enhanced info
        st.header(f"📊 AI Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Document", results['filename'])
        with col2:
            st.metric("📊 File Size", f"{results['file_size'] / 1024:.1f} KB")
        with col3:
            st.metric("⏰ Analyzed", results['analysis_timestamp'])
        
        st.divider()
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Executive Summary", 
            "🔍 Key Information", 
            "📝 Clause Analysis", 
            "📖 Simplified Document",
            "📥 Download Report"
        ])
        
        with tab1:
            st.subheader("📋 Executive Summary")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(results['summary'])
            
            with col2:
                st.metric("📂 Document Type", results['document_type'])
                st.metric("📝 Word Count", len(results['original_text'].split()))
                st.metric("📄 Clauses Found", len(results['clauses']))
                
                # Analysis quality indicator
                if "Watson NLU + Granite" in results['summary']:
                    st.success("🚀 Premium AI Analysis")
                elif "Watson NLU" in results['summary']:
                    st.info("⚡ Enhanced Analysis")
                else:
                    st.warning("⚙️ Standard Analysis")
        
        with tab2:
            st.subheader("🔍 Extracted Key Information")
            
            entities = results['entities']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if entities['parties'] and not any('Could not extract' in str(p) for p in entities['parties']):
                    st.markdown("**👥 Parties Involved:**")
                    for party in entities['parties']:
                        st.write(f"• {party}")
                else:
                    st.info("👥 Parties: Could not extract specific party information")
                
                if entities['dates'] and not any('Could not extract' in str(d) for d in entities['dates']):
                    st.markdown("**📅 Important Dates:**")
                    for date in entities['dates']:
                        st.write(f"• {date}")
                else:
                    st.info("📅 Dates: Could not extract specific date information")
            
            with col2:
                if entities['monetary_values'] and not any('Could not extract' in str(m) for m in entities['monetary_values']):
                    st.markdown("**💰 Financial Terms:**")
                    for value in entities['monetary_values']:
                        st.write(f"• {value}")
                else:
                    st.info("💰 Financial Terms: Could not extract specific monetary information")
                
                if entities['legal_terms'] and not any('Could not extract' in str(t) for t in entities['legal_terms']):
                    st.markdown("**⚖️ Key Legal Terms:**")
                    for term in entities['legal_terms'][:8]:  # Show more terms
                        st.write(f"• {term}")
                else:
                    st.info("⚖️ Legal Terms: Could not extract specific legal terminology")
            
            if entities['obligations'] and not any('Could not extract' in str(o) for o in entities['obligations']):
                st.markdown("**📋 Key Obligations:**")
                for obligation in entities['obligations'][:3]:
                    st.info(f"📌 {obligation}")
        
        with tab3:
            st.subheader("📝 AI-Powered Clause Analysis")
            
            if results['clauses'] and results['clause_explanations']:
                for i, (clause, explanation) in enumerate(zip(results['clauses'], results['clause_explanations'])):
                    with st.expander(f"📄 Clause {i+1}: {explanation[:60]}..."):
                        st.markdown("**🔤 Original Legal Text:**")
                        st.text_area("", clause, height=100, key=f"clause_{i}", disabled=True)
                        
                        st.markdown("**💡 AI Explanation (Plain English):**")
                        st.success(explanation)
                        
                        # Clause importance indicator
                        importance_keywords = ['termination', 'liability', 'payment', 'confidential', 'breach']
                        if any(keyword in clause.lower() for keyword in importance_keywords):
                            st.warning("⚠️ This appears to be a critical clause requiring careful attention")
            else:
                st.info("Could not extract individual clauses from this document. Please check the simplified document tab for the complete analysis.")
        
        with tab4:
            st.subheader("📖 AI-Simplified Document")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**🤖 Granite AI Simplified Version:**")
                with st.container(height=500):
                    st.markdown(results['simplified_text'])
            
            with col2:
                st.markdown("**📊 Simplification Metrics:**")
                original_words = len(results['original_text'].split())
                simplified_words = len(results['simplified_text'].split())
                
                st.metric("📝 Original Words", original_words)
                st.metric("✨ Simplified Words", simplified_words)
                
                if simplified_words != original_words:
                    if simplified_words < original_words:
                        reduction = ((original_words - simplified_words) / original_words) * 100
                        st.metric("📉 Complexity Reduction", f"{reduction:.1f}%")
                    else:
                        expansion = ((simplified_words - original_words) / original_words) * 100
                        st.metric("📈 Content Expansion", f"{expansion:.1f}%")
                
                st.markdown("---")
                st.markdown("**🎯 Readability:**")
                if "IBM Watson NLU + Granite" in results['summary']:
                    st.success("🚀 Premium AI Enhanced")
                elif "Granite Model" in results['summary']:
                    st.success("⚡ AI Enhanced")
                else:
                    st.info("⚙️ Rule-based")
        
        with tab5:
            st.subheader("📥 Generate Analysis Report")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.info("📋 Generate a comprehensive PDF report of your AI-powered document analysis")
                
                if st.button("📄 Generate Premium PDF Report", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is generating your comprehensive report..."):
                        try:
                            pdf_bytes = pdf_generator.generate_analysis_report(
                                results['original_text'],
                                results['simplified_text'],
                                results['clauses'],
                                results['clause_explanations'],
                                results['entities'],
                                results['document_type'],
                                results['summary']
                            )
                            
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"ClauseWise_AI_Analysis_{timestamp}.pdf"
                            
                            st.download_button(
                                label="📥 Download AI Analysis Report",
                                data=pdf_bytes,
                                file_name=filename,
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            st.success("✅ Premium AI report generated successfully!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error generating PDF report: {str(e)}")
                            st.info("💡 Please try again or contact support if the issue persists.")
    
    else:
        # Welcome screen when no document is uploaded
        st.markdown("""
        ### 🚀 Welcome to ClauseWise AI Legal Analyzer
        
        **Powered by IBM Watson & Granite AI Models**
        
        📋 **What ClauseWise Can Do:**
        - 🤖 **AI Document Classification** - Automatically identify contract types
        - 📝 **Intelligent Text Simplification** - Convert complex legal jargon to plain English  
        - 🔍 **Advanced Entity Recognition** - Extract parties, dates, financial terms
        - 💡 **Smart Clause Explanation** - Get AI-powered explanations for each clause
        - 📊 **Comprehensive Analysis** - Generate detailed insights and summaries
        - 📄 **Professional Reports** - Download analysis as PDF reports
        
        **🎯 Perfect for:** Lawyers, Business Professionals, Students, Anyone dealing with legal documents
        
        **⬅️ Get started by uploading a document in the sidebar!**
        """)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🤖 **AI-Powered**\n\nUtilizes IBM Watson NLU and Granite models for advanced analysis")
        with col2:
            st.info("📊 **Comprehensive**\n\nExtracts entities, classifies documents, and explains clauses")
        with col3:
            st.info("📄 **Professional**\n\nGenerate detailed PDF reports for your analysis")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 14px;'>
            <p><strong>ClauseWise Legal Document Analyzer v2.0</strong> | Powered by IBM Watson & Granite AI</p>
            <p><small>⚠️ This tool provides AI-generated analysis for informational purposes only. Always consult qualified legal counsel for legal decisions.</small></p>
            <p><small>🔐 Your documents are processed securely using enterprise-grade IBM AI services.</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()