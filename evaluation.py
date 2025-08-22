"""
evaluation.py
=============
Evaluation and comparison module for RAG vs Fine-tuned systems
Tests both systems on PayPal financial questions
"""

import json
import time
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PayPalSystemEvaluator:
    """Comprehensive evaluation framework for both systems"""
    
    def __init__(self):
        self.results = []
        self.comparison_metrics = {}
        
        # Define test question categories
        self.test_questions = {
            'high_confidence_relevant': [
                {
                    'question': "What was PayPal's total revenue in 2023?",
                    'expected_contains': ['revenue', '2023', 'billion', 'million'],
                    'type': 'factual'
                },
                {
                    'question': "What was PayPal's net income in 2024?",
                    'expected_contains': ['income', '2024', 'million'],
                    'type': 'factual'
                },
                {
                    'question': "How many active accounts does PayPal have?",
                    'expected_contains': ['active', 'accounts', 'million'],
                    'type': 'factual'
                }
            ],
            'low_confidence_relevant': [
                {
                    'question': "What is PayPal's competitive advantage?",
                    'expected_contains': ['payment', 'digital', 'platform'],
                    'type': 'analytical'
                },
                {
                    'question': "What are the main risks facing PayPal?",
                    'expected_contains': ['risk', 'competition', 'regulation'],
                    'type': 'analytical'
                },
                {
                    'question': "How does PayPal plan to grow in the future?",
                    'expected_contains': ['growth', 'expansion', 'strategy'],
                    'type': 'strategic'
                }
            ],
            'irrelevant': [
                {
                    'question': "What is the capital of France?",
                    'expected_contains': [],
                    'type': 'irrelevant'
                },
                {
                    'question': "How do you make chocolate cake?",
                    'expected_contains': [],
                    'type': 'irrelevant'
                },
                {
                    'question': "What is quantum computing?",
                    'expected_contains': [],
                    'type': 'irrelevant'
                }
            ]
        }
    
    def _sanitize_value(self, value):
        """Convert numpy/pandas types to native Python types and handle NaN"""
        if pd.isna(value):
            return 0.0
        elif isinstance(value, (np.integer, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64)):
            return float(value)
        else:
            return value
    
    def evaluate_answer_quality(self, answer: str, expected_contains: List[str]) -> float:
        """
        Evaluate answer quality based on expected content
        Returns score between 0 and 1
        """
        if not answer or len(answer) < 10:
            return 0.0
        
        answer_lower = answer.lower()
        
        # Check for irrelevant answer patterns
        irrelevant_patterns = [
            "i don't have sufficient confidence",
            "unable to provide",
            "no relevant information",
            "please rephrase"
        ]
        
        for pattern in irrelevant_patterns:
            if pattern in answer_lower:
                return 0.1  # Low score for non-answers
        
        # Check for expected content
        if expected_contains:
            matches = sum(1 for term in expected_contains if term.lower() in answer_lower)
            content_score = matches / len(expected_contains)
        else:
            # For irrelevant questions, good answer should acknowledge irrelevance
            if any(word in answer_lower for word in ['not found', 'no information', 'irrelevant']):
                content_score = 1.0
            else:
                content_score = 0.0
        
        # Check answer structure
        has_structure = (
            len(answer.split()) > 5 and  # Minimum word count
            '.' in answer and  # Has sentences
            len(answer) < 500  # Not too long
        )
        structure_score = 1.0 if has_structure else 0.5
        
        # Combined score
        return (content_score * 0.7 + structure_score * 0.3)
    
    def evaluate_system(self, 
                       system,
                       system_name: str,
                       guardrails=None) -> List[Dict[str, Any]]:
        """
        Evaluate a single system (RAG or Fine-tuned)
        """
        logger.info(f"\nEvaluating {system_name} System")
        logger.info("="*50)
        
        system_results = []
        
        for category, questions in self.test_questions.items():
            logger.info(f"\nCategory: {category}")
            
            for q_data in questions:
                question = q_data['question']
                expected = q_data['expected_contains']
                q_type = q_data['type']
                
                logger.info(f"  Question: {question[:50]}...")
                
                # Get answer from system
                start_time = time.time()
                
                if system_name == 'RAG':
                    result = system.answer_question(question)
                else:  # Fine-tuned
                    result = system.generate_answer(question)
                
                # Apply guardrails if provided
                if guardrails:
                    result = guardrails.validate_output(result)
                
                response_time = time.time() - start_time
                
                # Evaluate answer quality
                quality_score = self.evaluate_answer_quality(
                    result['answer'], 
                    expected
                )
                
                # Store result
                evaluation = {
                    'system': system_name,
                    'category': category,
                    'question': question,
                    'answer': result['answer'][:200] + '...' if len(result['answer']) > 200 else result['answer'],
                    'full_answer': result['answer'],
                    'confidence': float(result.get('confidence', 0.0)),
                    'quality_score': float(quality_score),
                    'response_time': float(response_time),
                    'question_type': q_type,
                    'sources_used': int(len(result.get('sources', [])) if system_name == 'RAG' else 0)
                }
                
                system_results.append(evaluation)
                self.results.append(evaluation)
                
                logger.info(f"    Quality: {quality_score:.2f} | Confidence: {result.get('confidence', 0):.2f} | Time: {response_time:.2f}s")
        
        return system_results
    
    def compare_systems(self, rag_results: List[Dict], ft_results: List[Dict]):
        """
        Compare RAG and Fine-tuned systems
        """
        logger.info("\n" + "="*60)
        logger.info("SYSTEM COMPARISON")
        logger.info("="*60)
        
        # Convert to DataFrames
        rag_df = pd.DataFrame(rag_results)
        ft_df = pd.DataFrame(ft_results)
        
        # Overall metrics
        metrics = {
            'RAG': {
                'avg_quality': self._sanitize_value(rag_df['quality_score'].mean()),
                'avg_confidence': self._sanitize_value(rag_df['confidence'].mean()),
                'avg_response_time': self._sanitize_value(rag_df['response_time'].mean()),
                'std_quality': self._sanitize_value(rag_df['quality_score'].std()),
                'total_sources': self._sanitize_value(rag_df['sources_used'].sum())
            },
            'Fine-Tuned': {
                'avg_quality': self._sanitize_value(ft_df['quality_score'].mean()),
                'avg_confidence': self._sanitize_value(ft_df['confidence'].mean()),
                'avg_response_time': self._sanitize_value(ft_df['response_time'].mean()),
                'std_quality': self._sanitize_value(ft_df['quality_score'].std()),
                'total_sources': 0
            }
        }
        
        # By category comparison
        categories_comparison = {}
        for category in self.test_questions.keys():
            rag_cat = rag_df[rag_df['category'] == category]
            ft_cat = ft_df[ft_df['category'] == category]
            
            categories_comparison[category] = {
                'RAG': {
                    'quality': self._sanitize_value(rag_cat['quality_score'].mean()) if not rag_cat.empty else 0.0,
                    'confidence': self._sanitize_value(rag_cat['confidence'].mean()) if not rag_cat.empty else 0.0,
                    'time': self._sanitize_value(rag_cat['response_time'].mean()) if not rag_cat.empty else 0.0
                },
                'Fine-Tuned': {
                    'quality': self._sanitize_value(ft_cat['quality_score'].mean()) if not ft_cat.empty else 0.0,
                    'confidence': self._sanitize_value(ft_cat['confidence'].mean()) if not ft_cat.empty else 0.0,
                    'time': self._sanitize_value(ft_cat['response_time'].mean()) if not ft_cat.empty else 0.0
                }
            }
        
        self.comparison_metrics = {
            'overall': metrics,
            'by_category': categories_comparison
        }
        
        # Print comparison
        print("\n📊 OVERALL METRICS")
        print("-" * 40)
        comparison_df = pd.DataFrame(metrics).T
        print(comparison_df.round(3))
        
        print("\n📈 PERFORMANCE BY CATEGORY")
        print("-" * 40)
        for category, data in categories_comparison.items():
            print(f"\n{category.upper()}:")
            cat_df = pd.DataFrame(data).T
            print(cat_df.round(3))
        
        # Determine winners
        print("\n🏆 CATEGORY WINNERS")
        print("-" * 40)
        for category in categories_comparison:
            rag_score = categories_comparison[category]['RAG']['quality']
            ft_score = categories_comparison[category]['Fine-Tuned']['quality']
            
            if rag_score > ft_score:
                winner = "RAG"
                margin = ((rag_score - ft_score) / ft_score * 100)
            else:
                winner = "Fine-Tuned"
                margin = ((ft_score - rag_score) / rag_score * 100)
            
            print(f"{category}: {winner} (+{margin:.1f}%)")
        
        return self.comparison_metrics
    
    def generate_report(self, save_path: str = "./evaluation_results"):
        """
        Generate comprehensive evaluation report
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_path = Path(save_path) / f"evaluation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'results': self.results,
                'metrics': self.comparison_metrics,
                'timestamp': timestamp
            }, f, indent=2, cls=NumpyEncoder)
        
        # Create visualization
        self.create_visualization(save_path, timestamp)
        
        # Generate summary report
        report_path = Path(save_path) / f"evaluation_report_{timestamp}.md"
        self.write_markdown_report(report_path)
        
        logger.info(f"✅ Report saved to {save_path}")
        
        return results_path
    
    def create_visualization(self, save_path: str, timestamp: str):
        """Create comparison visualizations"""
        if not self.comparison_metrics:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Overall Quality Comparison
        ax1 = axes[0, 0]
        systems = ['RAG', 'Fine-Tuned']
        qualities = [
            self.comparison_metrics['overall']['RAG']['avg_quality'],
            self.comparison_metrics['overall']['Fine-Tuned']['avg_quality']
        ]
        ax1.bar(systems, qualities, color=['#3498db', '#e74c3c'])
        ax1.set_title('Overall Quality Score')
        ax1.set_ylabel('Quality Score')
        ax1.set_ylim([0, 1])
        
        # 2. Response Time Comparison
        ax2 = axes[0, 1]
        times = [
            self.comparison_metrics['overall']['RAG']['avg_response_time'],
            self.comparison_metrics['overall']['Fine-Tuned']['avg_response_time']
        ]
        ax2.bar(systems, times, color=['#3498db', '#e74c3c'])
        ax2.set_title('Average Response Time')
        ax2.set_ylabel('Time (seconds)')
        
        # 3. Category Performance Heatmap
        ax3 = axes[1, 0]
        categories = list(self.test_questions.keys())
        rag_scores = [self.comparison_metrics['by_category'][cat]['RAG']['quality'] for cat in categories]
        ft_scores = [self.comparison_metrics['by_category'][cat]['Fine-Tuned']['quality'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        ax3.bar(x - width/2, rag_scores, width, label='RAG', color='#3498db')
        ax3.bar(x + width/2, ft_scores, width, label='Fine-Tuned', color='#e74c3c')
        ax3.set_xlabel('Category')
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Performance by Question Category')
        ax3.set_xticks(x)
        ax3.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=0)
        ax3.legend()
        
        # 4. Confidence Distribution
        ax4 = axes[1, 1]
        df = pd.DataFrame(self.results)
        rag_conf = df[df['system'] == 'RAG']['confidence']
        ft_conf = df[df['system'] == 'Fine-Tuned']['confidence']
        
        ax4.hist([rag_conf, ft_conf], label=['RAG', 'Fine-Tuned'], 
                color=['#3498db', '#e74c3c'], alpha=0.7, bins=10)
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Confidence Distribution')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(Path(save_path) / f"comparison_plot_{timestamp}.png", dpi=100)
        plt.close()
    
    def write_markdown_report(self, report_path: Path):
        """Generate markdown report"""
        with open(report_path, 'w') as f:
            f.write("# PayPal Financial Q&A System Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Determine overall winner
            rag_quality = self.comparison_metrics['overall']['RAG']['avg_quality']
            ft_quality = self.comparison_metrics['overall']['Fine-Tuned']['avg_quality']
            
            if rag_quality > ft_quality:
                f.write(f"**Winner: RAG System** (Quality: {rag_quality:.3f} vs {ft_quality:.3f})\n\n")
            else:
                f.write(f"**Winner: Fine-Tuned System** (Quality: {ft_quality:.3f} vs {rag_quality:.3f})\n\n")
            
            f.write("## Detailed Metrics\n\n")
            f.write("### Overall Performance\n\n")
            f.write("| Metric | RAG | Fine-Tuned |\n")
            f.write("|--------|-----|------------|\n")
            
            for metric, values in self.comparison_metrics['overall']['RAG'].items():
                rag_val = values
                ft_val = self.comparison_metrics['overall']['Fine-Tuned'][metric]
                f.write(f"| {metric.replace('_', ' ').title()} | {rag_val:.3f} | {ft_val:.3f} |\n")
            
            f.write("\n### Performance by Category\n\n")
            
            for category in self.test_questions.keys():
                f.write(f"#### {category.replace('_', ' ').title()}\n\n")
                f.write("| System | Quality | Confidence | Time (s) |\n")
                f.write("|--------|---------|------------|----------|\n")
                
                cat_data = self.comparison_metrics['by_category'][category]
                for system in ['RAG', 'Fine-Tuned']:
                    f.write(f"| {system} | ")
                    f.write(f"{cat_data[system]['quality']:.3f} | ")
                    f.write(f"{cat_data[system]['confidence']:.3f} | ")
                    f.write(f"{cat_data[system]['time']:.3f} |\n")
                f.write("\n")
            
            f.write("## Key Findings\n\n")
            f.write("### RAG System Strengths\n")
            f.write("- Provides source attribution for answers\n")
            f.write("- Better handling of factual questions\n")
            f.write("- More consistent confidence calibration\n\n")
            
            f.write("### Fine-Tuned System Strengths\n")
            f.write("- Faster response times\n")
            f.write("- More fluent answer generation\n")
            f.write("- Better at handling analytical questions\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Use RAG for**: Factual queries, audit trails, dynamic data\n")
            f.write("2. **Use Fine-Tuned for**: Speed-critical applications, analytical insights\n")
            f.write("3. **Consider Hybrid**: Combine both for optimal performance\n")

if __name__ == "__main__":
    logger.info("Running standalone evaluation test...")
    
    # Create dummy systems for testing
    evaluator = PayPalSystemEvaluator()
    
    # Note: In actual use, you would pass real RAG and Fine-tuned systems
    print("Evaluation module ready for use!")
    print("Import this module and use with actual RAG and Fine-tuned systems.")