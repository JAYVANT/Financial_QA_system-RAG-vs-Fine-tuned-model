"""
rag_system.py
=============
RAG (Retrieval-Augmented Generation) system for PayPal reports
Implements hybrid retrieval, re-ranking, and answer generation
"""

import json
import time
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PayPalRAGSystem:
    """RAG system specifically tuned for PayPal financial reports"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator_model: str = "gpt2"):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info("Loading cross-encoder...")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        logger.info("Loading generator model...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(generator_model)
        self.generator = GPT2LMHeadModel.from_pretrained(generator_model).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize retrieval components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.dense_index = None
        self.sparse_matrix = None
        self.chunks = []
        
        # Retrieval parameters
        self.top_k_dense = 5
        self.top_k_sparse = 5
        self.rerank_top_k = 3
        
    def build_indices(self, chunks: List[Dict[str, Any]]):
        """
        Build both dense (FAISS) and sparse (TF-IDF) indices
        """
        logger.info(f"Building indices for {len(chunks)} chunks...")
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Build dense index
        logger.info("Creating dense embeddings...")
        embeddings = self.embedding_model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
        self.dense_index.add(embeddings)
        
        # Build sparse index
        logger.info("Creating sparse index...")
        self.sparse_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info(f"✅ Indices built successfully!")
        
    def hybrid_retrieve(self, query: str, filter_year: str = None) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining dense and sparse methods
        """
        # Dense retrieval
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        dense_scores, dense_indices = self.dense_index.search(
            query_embedding, 
            min(self.top_k_dense * 2, len(self.chunks))  # Get more for filtering
        )
        
        # Sparse retrieval
        query_vector = self.tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_vector, self.sparse_matrix).flatten()
        sparse_top_indices = np.argsort(sparse_scores)[::-1][:self.top_k_sparse * 2]
        
        # Combine results
        results = {}
        
        # Process dense results
        for idx, score in zip(dense_indices[0], dense_scores[0]):
            chunk = self.chunks[idx]
            
            # Apply year filter if specified
            if filter_year and chunk['metadata'].get('year') != filter_year:
                continue
                
            chunk_id = chunk['chunk_id']
            results[chunk_id] = {
                'chunk': chunk,
                'dense_score': float(score),
                'sparse_score': 0.0,
                'combined_score': float(score) * 0.7  # Weight for dense
            }
        
        # Process sparse results
        for idx in sparse_top_indices:
            chunk = self.chunks[idx]
            
            # Apply year filter
            if filter_year and chunk['metadata'].get('year') != filter_year:
                continue
                
            chunk_id = chunk['chunk_id']
            score = float(sparse_scores[idx])
            
            if chunk_id in results:
                # Combine scores if already retrieved
                results[chunk_id]['sparse_score'] = score
                results[chunk_id]['combined_score'] += score * 0.3  # Weight for sparse
            else:
                results[chunk_id] = {
                    'chunk': chunk,
                    'dense_score': 0.0,
                    'sparse_score': score,
                    'combined_score': score * 0.3
                }
        
        # Sort by combined score
        sorted_results = sorted(
            results.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )[:self.top_k_dense + self.top_k_sparse]
        
        return sorted_results
    
    def rerank_with_cross_encoder(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank retrieved chunks using cross-encoder for better accuracy
        """
        if not results:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [[query, result['chunk']['text']] for result in results]
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Update scores
        for i, result in enumerate(results):
            result['ce_score'] = float(ce_scores[i])
            # Combine retrieval and cross-encoder scores
            result['final_score'] = (
                result['combined_score'] * 0.3 + 
                result['ce_score'] * 0.7
            )
        
        # Sort by final score
        reranked = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        return reranked[:self.rerank_top_k]
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using retrieved context from PayPal reports
        """
        start_time = time.time()
        
        # Prepare context from retrieved chunks
        context_parts = []
        for i, result in enumerate(retrieved_chunks):
            chunk = result['chunk']
            year = chunk['metadata'].get('year', 'Unknown')
            section = chunk['metadata'].get('section', 'Unknown')
            context_parts.append(
                f"[{year} - {section}]: {chunk['text'][:300]}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on PayPal's annual reports, answer the following question.

Context from reports:
{context}

Question: {query}

Answer based on the context above:"""
        
        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=800,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.generator.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer based on the context above:" in full_response:
            answer = full_response.split("Answer based on the context above:")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()
        
        # Calculate confidence based on retrieval scores
        avg_score = np.mean([r['final_score'] for r in retrieved_chunks]) if retrieved_chunks else 0
        
        # Prepare source citations
        sources = []
        for result in retrieved_chunks:
            chunk = result['chunk']
            sources.append({
                'year': chunk['metadata'].get('year'),
                'section': chunk['metadata'].get('section'),
                'score': result['final_score']
            })
        
        return {
            'answer': answer,
            'confidence': float(avg_score),
            'time': time.time() - start_time,
            'sources': sources,
            'num_chunks_used': len(retrieved_chunks),
            'method': 'RAG'
        }
    
    def answer_question(self, query: str, year_filter: str = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve -> rerank -> generate
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        retrieved = self.hybrid_retrieve(query, filter_year=year_filter)
        
        if not retrieved:
            return {
                'answer': "No relevant information found in the PayPal reports.",
                'confidence': 0.0,
                'time': 0.0,
                'sources': [],
                'method': 'RAG'
            }
        
        # Rerank with cross-encoder
        reranked = self.rerank_with_cross_encoder(query, retrieved)
        
        # Generate answer
        result = self.generate_answer(query, reranked)
        
        logger.info(f"Answer generated in {result['time']:.2f}s with confidence {result['confidence']:.2f}")
        
        return result

class RAGGuardrails:
    """Input and output guardrails for RAG system"""
    
    def __init__(self):
        self.min_confidence_threshold = 0.3
        self.max_answer_length = 500
        
        # PayPal-specific validation
        self.valid_topics = [
            'revenue', 'income', 'profit', 'loss', 'expense',
            'assets', 'liabilities', 'cash', 'transactions',
            'users', 'accounts', 'growth', 'payment', 'volume',
            'paypal', 'venmo', 'braintree', 'digital', 'wallet'
        ]
        
        self.invalid_patterns = [
            'password', 'hack', 'exploit', 'confidential',
            'secret', 'private key', 'api key'
        ]
    
    def validate_input(self, query: str) -> Tuple[bool, str]:
        """Validate input query"""
        query_lower = query.lower()
        
        # Check for invalid patterns
        for pattern in self.invalid_patterns:
            if pattern in query_lower:
                return False, f"Query contains restricted term: {pattern}"
        
        # Check query length
        if len(query) < 5:
            return False, "Query too short"
        if len(query) > 500:
            return False, "Query too long"
        
        # Check if query is somewhat relevant (soft check)
        has_relevant_term = any(topic in query_lower for topic in self.valid_topics)
        is_question = any(q in query_lower for q in ['what', 'how', 'when', 'where', 'why', 'which'])
        
        if not (has_relevant_term or is_question):
            # Still allow but flag as potentially off-topic
            pass
        
        return True, "Valid query"
    
    def validate_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and potentially modify output"""
        # Check confidence
        if response['confidence'] < self.min_confidence_threshold:
            response['answer'] = (
                "I don't have sufficient confidence to answer this question based on "
                "the available PayPal report data. Please rephrase or ask about specific "
                "financial metrics, revenue, users, or business segments."
            )
            response['low_confidence'] = True
        
        # Check answer length
        if len(response['answer']) > self.max_answer_length:
            response['answer'] = response['answer'][:self.max_answer_length] + "..."
            response['truncated'] = True
        
        # Check for empty or nonsensical answers
        if len(response['answer']) < 10 or response['answer'].count(' ') < 2:
            response['answer'] = "Unable to generate a meaningful answer. Please try rephrasing your question."
            response['generation_error'] = True
        
        return response

def load_and_initialize_rag(processed_data_path: str = "./processed_data/paypal_processed_data.json"):
    """
    Load processed data and initialize RAG system
    """
    logger.info("Initializing RAG system...")
    
    # Load processed data
    with open(processed_data_path, 'r') as f:
        data = json.load(f)
    
    # Initialize RAG system
    rag = PayPalRAGSystem()
    
    # Build indices from chunks
    rag.build_indices(data['chunks'])
    
    # Initialize guardrails
    guardrails = RAGGuardrails()
    
    logger.info("✅ RAG system ready!")
    
    return rag, guardrails, data

if __name__ == "__main__":
    # Load and test RAG system
    rag, guardrails, data = load_and_initialize_rag()
    
    # Test questions
    test_questions = [
        "What was PayPal's revenue in 2023?",
        "How did PayPal's revenue change from 2023 to 2024?",
        "What are PayPal's main business segments?",
        "What was the total payment volume in 2024?",
        "How many active accounts does PayPal have?"
    ]
    
    print("\n" + "="*60)
    print("Testing RAG System with PayPal Reports")
    print("="*60)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        
        # Validate input
        is_valid, message = guardrails.validate_input(question)
        if not is_valid:
            print(f"❌ Invalid query: {message}")
            continue
        
        # Get answer
        result = rag.answer_question(question)
        
        # Validate output
        result = guardrails.validate_output(result)
        
        # Display result
        print(f"📝 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"⏱️ Time: {result['time']:.2f}s")
        print(f"📚 Sources: {len(result['sources'])} chunks used")