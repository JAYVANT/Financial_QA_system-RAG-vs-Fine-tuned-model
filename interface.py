"""
interface.py
============
Streamlit web interface for PayPal Financial Q&A System
Provides interactive UI for testing both RAG and Fine-tuned systems
"""

import streamlit as st
import json
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import our modules
from rag_system import load_and_initialize_rag
from finetune_system import PayPalFineTunedModel, FineTuneGuardrails

# Page configuration
st.set_page_config(
    page_title="PayPal Financial Q&A System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0070ba;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0070ba;
    }
</style>
""", unsafe_allow_html=True)

class PayPalQAInterface:
    """Streamlit interface for PayPal Q&A System"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'questions_history' not in st.session_state:
            st.session_state.questions_history = []
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = []
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = 'compare'
    
    def display_header(self):
        """Display application header"""
        st.markdown("<h1 class='main-header'>💰 PayPal Financial Q&A System</h1>", 
                   unsafe_allow_html=True)
        st.markdown("### Compare RAG vs Fine-Tuned Models on PayPal Annual Reports (2023-2024)")
        st.markdown("---")
    
    def display_sidebar(self):
        """Display sidebar controls"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Mode selection
            mode = st.selectbox(
                "Query Mode",
                ["compare", "rag", "fine-tuned"],
                format_func=lambda x: x.replace('-', ' ').title()
            )
            st.session_state.current_mode = mode
            
            # Year filter for RAG
            if mode in ['rag', 'compare']:
                st.subheader("RAG Settings")
                year_filter = st.selectbox(
                    "Filter by Year",
                    ["All", "2023", "2024"]
                )
                st.session_state.year_filter = None if year_filter == "All" else year_filter
            
            # Sample questions
            st.subheader("📝 Sample Questions")
            sample_questions = [
                "What was PayPal's total revenue in 2023?",
                "What was PayPal's net income in 2024?",
                "How did revenue change from 2023 to 2024?",
                "What are PayPal's main business segments?",
                "How many active accounts does PayPal have?",
                "What is PayPal's total payment volume?",
                "What are the key risks facing PayPal?",
                "What is PayPal's growth strategy?"
            ]
            
            selected_question = st.selectbox(
                "Select a sample question:",
                ["Custom"] + sample_questions
            )
            
            if selected_question != "Custom":
                st.session_state.selected_question = selected_question
            
            # System metrics
            if hasattr(self, 'systems') and self.systems:
                st.subheader("📊 System Status")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RAG Chunks", len(self.systems['data']['chunks']))
                with col2:
                    st.metric("Q&A Pairs", len(self.systems['data']['qa_pairs']))
            
            # History
            if st.session_state.questions_history:
                st.subheader("📜 Recent Questions")
                for q in st.session_state.questions_history[-5:]:
                    st.text(f"• {q[:30]}...")
    
    def process_question(self, question: str):
        """Process question through selected system(s)"""
        results = {}
        
        # RAG System
        if st.session_state.current_mode in ['rag', 'compare']:
            with st.spinner("🔍 RAG system processing..."):
                year_filter = getattr(st.session_state, 'year_filter', None)
                rag_result = self.systems['rag'].answer_question(question, year_filter)
                rag_result = self.systems['rag_guardrails'].validate_output(rag_result)
                results['rag'] = rag_result
        
        # Fine-Tuned System
        if st.session_state.current_mode in ['fine-tuned', 'compare']:
            with st.spinner("🧠 Fine-tuned model processing..."):
                ft_result = self.systems['ft'].generate_answer(question)
                ft_result = self.systems['ft_guardrails'].validate_output(ft_result)
                results['ft'] = ft_result
        
        # Add to history
        if question not in st.session_state.questions_history:
            st.session_state.questions_history.append(question)
        
        # Store comparison if both systems used
        if 'rag' in results and 'ft' in results:
            st.session_state.comparison_results.append({
                'question': question,
                'rag_time': results['rag']['time'],
                'ft_time': results['ft']['time'],
                'rag_confidence': results['rag']['confidence'],
                'ft_confidence': results['ft']['confidence']
            })
        
        return results
    
    def display_results(self, results: dict, question: str):
        """Display results from system(s)"""
        st.subheader(f"❓ Question: {question}")
        
        if st.session_state.current_mode == 'compare':
            # Side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔍 RAG System")
                self.display_single_result(results.get('rag'), 'rag')
            
            with col2:
                st.markdown("### 🧠 Fine-Tuned Model")
                self.display_single_result(results.get('ft'), 'ft')
            
            # Comparison metrics
            st.markdown("### 📊 Comparison")
            self.display_comparison(results)
            
        else:
            # Single system result
            system_name = "RAG" if st.session_state.current_mode == 'rag' else "Fine-Tuned"
            system_key = 'rag' if st.session_state.current_mode == 'rag' else 'ft'
            st.markdown(f"### {system_name} System")
            self.display_single_result(results.get(system_key), system_key)
    
    def display_single_result(self, result: dict, system: str):
        """Display single system result"""
        if not result:
            st.error("No result available")
            return
        
        # Answer
        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.write("**Answer:**")
        st.write(result['answer'])
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Metrics in a single row
        confidence_color = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.4 else "🔴"
        
        # Display metrics in a formatted way without nested columns
        st.write(f"**Confidence:** {confidence_color} {result['confidence']:.1%}")
        st.write(f"**Response Time:** {result['time']:.2f}s")
        
        if system == 'rag' and 'sources' in result:
            st.write(f"**Sources Used:** {len(result['sources'])}")
        else:
            st.write(f"**Method:** {result.get('method', 'Fine-Tuned')}")
        
        # Sources for RAG
        if system == 'rag' and 'sources' in result and result['sources']:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(result['sources'][:3]):
                    st.write(f"**Source {i+1}:**")
                    st.write(f"- Year: {source.get('year', 'N/A')}")
                    st.write(f"- Section: {source.get('section', 'N/A')}")
                    st.write(f"- Relevance: {source.get('score', 0):.2f}")
    
    def display_comparison(self, results: dict):
        """Display comparison visualization"""
        if 'rag' not in results or 'ft' not in results:
            return
        
        # Performance comparison chart
        fig = go.Figure(data=[
            go.Bar(name='RAG', x=['Confidence', 'Speed (1/time)'], 
                  y=[results['rag']['confidence'], 1/results['rag']['time']]),
            go.Bar(name='Fine-Tuned', x=['Confidence', 'Speed (1/time)'], 
                  y=[results['ft']['confidence'], 1/results['ft']['time']])
        ])
        fig.update_layout(
            title="Performance Comparison",
            barmode='group',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics table
        comparison_df = pd.DataFrame({
            'Metric': ['Confidence', 'Response Time', 'Answer Length'],
            'RAG': [
                f"{results['rag']['confidence']:.2%}",
                f"{results['rag']['time']:.2f}s",
                len(results['rag']['answer'])
            ],
            'Fine-Tuned': [
                f"{results['ft']['confidence']:.2%}",
                f"{results['ft']['time']:.2f}s",
                len(results['ft']['answer'])
            ]
        })
        st.dataframe(comparison_df, use_container_width=True)
    
    def display_analytics(self):
        """Display analytics dashboard"""
        if not st.session_state.comparison_results:
            st.info("No comparison data available yet. Ask some questions first!")
            return
        
        st.subheader("📈 Analytics Dashboard")
        
        df = pd.DataFrame(st.session_state.comparison_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average response times
            fig = go.Figure(data=[
                go.Bar(x=['RAG', 'Fine-Tuned'], 
                      y=[df['rag_time'].mean(), df['ft_time'].mean()],
                      marker_color=['#3498db', '#e74c3c'])
            ])
            fig.update_layout(
                title="Average Response Time",
                yaxis_title="Time (seconds)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average confidence
            fig = go.Figure(data=[
                go.Bar(x=['RAG', 'Fine-Tuned'], 
                      y=[df['rag_confidence'].mean(), df['ft_confidence'].mean()],
                      marker_color=['#3498db', '#e74c3c'])
            ])
            fig.update_layout(
                title="Average Confidence",
                yaxis_title="Confidence Score",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Questions table
        st.subheader("Question History")
        history_df = df[['question', 'rag_confidence', 'ft_confidence']].copy()
        history_df.columns = ['Question', 'RAG Confidence', 'FT Confidence']
        st.dataframe(history_df, use_container_width=True)
    
    def run(self):
        """Main application loop"""
        # Display header
        self.display_header()
        
        # Load systems
        self.systems = load_systems()
        
        if not self.systems:
            st.error("Failed to load systems. Please run `python main.py` first to initialize.")
            return
        
        # Display sidebar
        self.display_sidebar()
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["💬 Q&A", "📊 Analytics", "📚 Documentation"])
        
        with tab1:
            # Question input
            if hasattr(st.session_state, 'selected_question') and st.session_state.selected_question:
                question = st.text_area(
                    "Enter your question:",
                    value=st.session_state.selected_question,
                    height=80
                )
                st.session_state.selected_question = None
            else:
                question = st.text_area(
                    "Enter your question:",
                    placeholder="E.g., What was PayPal's revenue in 2023?",
                    height=80
                )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("🚀 Get Answer", type="primary"):
                    if question:
                        results = self.process_question(question)
                        self.display_results(results, question)
                    else:
                        st.warning("Please enter a question")
            
            with col2:
                if st.button("🔄 Clear"):
                    st.session_state.clear()
                    st.rerun()
        
        with tab2:
            self.display_analytics()
        
        with tab3:
            st.markdown("""
            ### 📖 System Documentation
            
            #### RAG System
            - **Retrieval**: Hybrid approach combining dense (semantic) and sparse (keyword) search
            - **Re-ranking**: Cross-encoder for improved accuracy
            - **Generation**: Context-aware answer generation with source attribution
            - **Best for**: Factual queries, audit trails, dynamic information
            
            #### Fine-Tuned System
            - **Architecture**: GPT-2 with LoRA adaptation
            - **Training**: Fine-tuned on PayPal Q&A pairs
            - **Optimization**: ~90% parameter reduction with LoRA
            - **Best for**: Speed-critical applications, analytical insights
            
            #### Data Sources
            - PayPal Annual Report 2023
            - PayPal Annual Report 2024
            - Processed into {chunks} chunks and {qa_pairs} Q&A pairs
            """.format(
                chunks=len(self.systems['data']['chunks']),
                qa_pairs=len(self.systems['data']['qa_pairs'])
            ))

@st.cache_resource
def load_systems():
    """Load both RAG and Fine-tuned systems"""
    try:
        # Load RAG system
        rag_system, rag_guardrails, processed_data = load_and_initialize_rag()
        
        # Load Fine-tuned system
        ft_system = PayPalFineTunedModel()
        model_path = "./models/paypal_finetuned"
        if Path(model_path).exists():
            ft_system.load_model(model_path)
        ft_guardrails = FineTuneGuardrails()
        
        return {
            'rag': rag_system,
            'rag_guardrails': rag_guardrails,
            'ft': ft_system,
            'ft_guardrails': ft_guardrails,
            'data': processed_data
        }
    except Exception as e:
        st.error(f"Error loading systems: {e}")
        return None

def main():
    """Main entry point for Streamlit app"""
    app = PayPalQAInterface()
    app.run()

if __name__ == "__main__":
    main()