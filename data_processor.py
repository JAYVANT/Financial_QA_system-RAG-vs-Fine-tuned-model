"""
data_processor.py
=================
Module for processing PayPal annual reports (2023 & 2024)
Extracts text, creates chunks, and generates Q&A pairs
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import PyPDF2
import pdfplumber
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PayPalReportProcessor:
    """Process PayPal annual reports for 2023 and 2024"""
    
    def __init__(self, data_dir: str = "./financial_data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path("./processed_data")
        self.processed_dir.mkdir(exist_ok=True)
        
        # PayPal-specific patterns
        self.paypal_patterns = {
            'revenue': [
                r'(?i)total\s+revenue[:\s]+\$?([\d,\.]+)\s*(billion|million)?',
                r'(?i)net\s+revenues?[:\s]+\$?([\d,\.]+)\s*(billion|million)?',
                r'(?i)revenue\s+was\s+\$?([\d,\.]+)\s*(billion|million)?'
            ],
            'income': [
                r'(?i)net\s+income[:\s]+\$?([\d,\.]+)\s*(billion|million)?',
                r'(?i)earnings?[:\s]+\$?([\d,\.]+)\s*(billion|million)?'
            ],
            'transactions': [
                r'(?i)payment\s+volume[:\s]+\$?([\d,\.]+)\s*(billion|million)?',
                r'(?i)total\s+payment\s+volume[:\s]+\$?([\d,\.]+)\s*(billion|million)?',
                r'(?i)([\d,\.]+)\s*(billion|million)?\s+transactions?'
            ],
            'users': [
                r'(?i)active\s+accounts?[:\s]+([\d,\.]+)\s*million',
                r'(?i)([\d,\.]+)\s*million\s+active\s+accounts?'
            ]
        }
    
    def extract_pdf_text(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text from PayPal PDF report
        Returns dict with page numbers as keys
        """
        logger.info(f"Extracting text from: {pdf_path}")
        pages_text = {}
        
        try:
            # Try pdfplumber first (better for tables)
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_text[f"page_{i+1}"] = text
                        
            # Fallback to PyPDF2 if needed
            if not pages_text:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            pages_text[f"page_{i+1}"] = text
            
            logger.info(f"Extracted {len(pages_text)} pages from {pdf_path}")
            return pages_text
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return {}
    
    def segment_report(self, pages_text: Dict[str, str]) -> Dict[str, str]:
        """
        Segment PayPal report into logical sections
        """
        sections = {
            'executive_summary': '',
            'financial_highlights': '',
            'income_statement': '',
            'balance_sheet': '',
            'cash_flow': '',
            'business_overview': '',
            'risk_factors': '',
            'management_discussion': ''
        }
        
        # Section identification patterns
        section_patterns = {
            'executive_summary': r'(?i)(executive\s+summary|letter\s+to\s+shareholders?|dear\s+shareholders?)',
            'financial_highlights': r'(?i)(financial\s+highlights?|selected\s+financial\s+data)',
            'income_statement': r'(?i)(consolidated\s+statements?\s+of\s+income|income\s+statements?)',
            'balance_sheet': r'(?i)(consolidated\s+balance\s+sheets?|statements?\s+of\s+financial\s+position)',
            'cash_flow': r'(?i)(consolidated\s+statements?\s+of\s+cash\s+flows?)',
            'business_overview': r'(?i)(business\s+overview|our\s+business|company\s+overview)',
            'risk_factors': r'(?i)(risk\s+factors?|risks?\s+relating)',
            'management_discussion': r'(?i)(management.s?\s+discussion|MD&A)'
        }
        
        # Combine all pages into one text
        full_text = '\n\n'.join(pages_text.values())
        
        # Try to identify and extract sections
        for section_name, pattern in section_patterns.items():
            matches = re.split(pattern, full_text, maxsplit=1)
            if len(matches) > 1:
                # Extract content after section header (limited to 15000 chars)
                section_content = matches[-1][:15000]
                # Clean up the content
                section_content = self.clean_text(section_content)
                sections[section_name] = section_content
                logger.info(f"Found section: {section_name} ({len(section_content)} chars)")
        
        # If no sections found, store full text
        if not any(sections.values()):
            sections['full_report'] = self.clean_text(full_text)
        
        return sections
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)\:\;\&]', '', text)
        return text.strip()
    
    def extract_financial_metrics(self, text: str, year: str) -> Dict[str, any]:
        """
        Extract specific PayPal financial metrics
        """
        metrics = {
            'year': year,
            'revenue': None,
            'net_income': None,
            'payment_volume': None,
            'active_accounts': None,
            'transactions': None
        }
        
        # Try to extract each metric
        for metric_type, patterns in self.paypal_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).replace(',', '')
                    unit = match.group(2) if len(match.groups()) > 1 else ''
                    
                    # Convert to standard format
                    try:
                        numeric_value = float(value)
                        if unit and 'billion' in unit.lower():
                            numeric_value *= 1000  # Convert to millions
                        
                        if metric_type == 'revenue':
                            metrics['revenue'] = f"${numeric_value:.0f}M"
                        elif metric_type == 'income':
                            metrics['net_income'] = f"${numeric_value:.0f}M"
                        elif metric_type == 'transactions':
                            metrics['payment_volume'] = f"${numeric_value:.0f}M"
                        elif metric_type == 'users':
                            metrics['active_accounts'] = f"{numeric_value:.0f}M"
                            
                    except ValueError:
                        continue
        
        return metrics
    
    def generate_qa_pairs(self, sections_2023: Dict[str, str], 
                         sections_2024: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs specific to PayPal reports
        """
        qa_pairs = []
        
        # Extract metrics for both years
        metrics_2023 = self.extract_financial_metrics(
            ' '.join(sections_2023.values()), '2023'
        )
        metrics_2024 = self.extract_financial_metrics(
            ' '.join(sections_2024.values()), '2024'
        )
        
        # Generate factual Q&A pairs
        qa_templates = [
            # 2023 specific
            {
                'question': "What was PayPal's total revenue in 2023?",
                'answer': f"PayPal's total revenue in 2023 was {metrics_2023.get('revenue', 'not found')}.",
                'year': '2023',
                'type': 'factual'
            },
            {
                'question': "What was PayPal's net income in 2023?",
                'answer': f"PayPal's net income in 2023 was {metrics_2023.get('net_income', 'not found')}.",
                'year': '2023',
                'type': 'factual'
            },
            # 2024 specific
            {
                'question': "What was PayPal's total revenue in 2024?",
                'answer': f"PayPal's total revenue in 2024 was {metrics_2024.get('revenue', 'not found')}.",
                'year': '2024',
                'type': 'factual'
            },
            {
                'question': "What was PayPal's net income in 2024?",
                'answer': f"PayPal's net income in 2024 was {metrics_2024.get('net_income', 'not found')}.",
                'year': '2024',
                'type': 'factual'
            },
            # Comparative questions
            {
                'question': "How did PayPal's revenue change from 2023 to 2024?",
                'answer': f"PayPal's revenue changed from {metrics_2023.get('revenue', 'N/A')} in 2023 to {metrics_2024.get('revenue', 'N/A')} in 2024.",
                'year': 'both',
                'type': 'comparative'
            },
            {
                'question': "What was the trend in PayPal's active accounts?",
                'answer': f"PayPal had {metrics_2023.get('active_accounts', 'N/A')} active accounts in 2023 and {metrics_2024.get('active_accounts', 'N/A')} in 2024.",
                'year': 'both',
                'type': 'trend'
            }
        ]
        
        # Add template Q&As
        qa_pairs.extend(qa_templates)
        
        # Generate section-specific questions
        for section_name in ['financial_highlights', 'income_statement', 'business_overview']:
            if sections_2023.get(section_name):
                # Extract key sentences for Q&A
                sentences = sections_2023[section_name].split('.')[:5]  # First 5 sentences
                for sentence in sentences:
                    if len(sentence) > 50 and any(word in sentence.lower() for word in ['revenue', 'income', 'growth', 'increased', 'decreased']):
                        qa_pairs.append({
                            'question': f"What does the 2023 report say about {section_name.replace('_', ' ')}?",
                            'answer': sentence.strip() + '.',
                            'year': '2023',
                            'type': 'descriptive'
                        })
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def create_chunks(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict[str, any]]:
        """
        Create overlapping text chunks for retrieval
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text) > 50:  # Minimum chunk size
                chunks.append({
                    'text': chunk_text,
                    'start_idx': i,
                    'end_idx': min(i + chunk_size, len(words)),
                    'chunk_id': f"chunk_{len(chunks)}",
                    'metadata': {
                        'chunk_size': chunk_size,
                        'word_count': len(chunk_words)
                    }
                })
        
        return chunks
    
    def process_reports(self, report_2023_path: str, report_2024_path: str) -> Dict:
        """
        Main processing function for both PayPal reports
        """
        logger.info("Starting PayPal report processing...")
        
        # Extract text from both reports
        pages_2023 = self.extract_pdf_text(report_2023_path)
        pages_2024 = self.extract_pdf_text(report_2024_path)
        
        # Segment reports
        sections_2023 = self.segment_report(pages_2023)
        sections_2024 = self.segment_report(pages_2024)
        
        # Generate Q&A pairs
        qa_pairs = self.generate_qa_pairs(sections_2023, sections_2024)
        
        # Create chunks for RAG
        all_chunks = []
        
        # Process 2023 chunks
        for section_name, section_text in sections_2023.items():
            if section_text:
                chunks = self.create_chunks(section_text, chunk_size=200)
                for chunk in chunks:
                    chunk['metadata']['year'] = '2023'
                    chunk['metadata']['section'] = section_name
                all_chunks.extend(chunks)
        
        # Process 2024 chunks
        for section_name, section_text in sections_2024.items():
            if section_text:
                chunks = self.create_chunks(section_text, chunk_size=200)
                for chunk in chunks:
                    chunk['metadata']['year'] = '2024'
                    chunk['metadata']['section'] = section_name
                all_chunks.extend(chunks)
        
        # Save processed data
        output = {
            'sections_2023': sections_2023,
            'sections_2024': sections_2024,
            'qa_pairs': qa_pairs,
            'chunks': all_chunks,
            'metadata': {
                'processed_date': datetime.now().isoformat(),
                'num_chunks': len(all_chunks),
                'num_qa_pairs': len(qa_pairs)
            }
        }
        
        # Save to file
        output_path = self.processed_dir / "paypal_processed_data.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Processing complete! Saved to {output_path}")
        logger.info(f"Created {len(all_chunks)} chunks and {len(qa_pairs)} Q&A pairs")
        
        return output

if __name__ == "__main__":
    # Process PayPal reports
    processor = PayPalReportProcessor()
    
    # Update these paths to match your file locations
    results = processor.process_reports(
        report_2023_path="./financial_data/Paypal2023_report.pdf",
        report_2024_path="./financial_data/Paypal2024_report.pdf"
    )
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Generated {len(results['chunks'])} chunks")
    print(f"❓ Created {len(results['qa_pairs'])} Q&A pairs")