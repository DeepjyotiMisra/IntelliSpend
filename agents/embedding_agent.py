"""
Embedding Agent - Generates embeddings for transaction data in IntelliSpend
This agent creates vector embeddings for merchants and transaction descriptions
to enable semantic similarity matching.
"""

from agno.agent import Agent
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import pickle
import os
from models.transaction import TransactionData

logger = logging.getLogger(__name__)


class EmbeddingAgent(Agent):
    """
    Agent responsible for generating and managing embeddings for transaction data.
    
    Key responsibilities:
    - Generate merchant embeddings
    - Create transaction description embeddings
    - Maintain embedding cache
    - Compute similarity scores
    - Update embedding models
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(
            name="EmbeddingAgent",
            description="Generates and analyzes embeddings for transaction data with quality monitoring and optimization tools",
            instructions=[
                "Generate high-quality embeddings for merchant names and transaction descriptions",
                "Maintain embedding cache for efficient processing",
                "Analyze embedding quality and detect anomalies",
                "Optimize batch processing for performance", 
                "Monitor embedding consistency and suggest improvements",
                "Compute semantic similarity scores for merchant matching"
            ],
            tools=[]  # No custom tools needed - all logic is in internal methods
        )
        self.model_name = model_name
        self.model = None
        self.embedding_cache = {}
        self.cache_file = "data/embeddings_cache.pkl"
        self._initialize_model()
        self._load_cache()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model with CPU device for compatibility."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Force CPU device to avoid MPS tensor issues
            import torch
            import os
            
            # Set environment variables to force CPU usage
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            device = 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Ensure model is on CPU
            self.model = self.model.to('cpu')
            
            logger.info(f"Embedding model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            # Fallback to a basic model with CPU
            try:
                import torch
                import os
                
                # Set environment variables to force CPU usage
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                device = 'cpu'
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                self.model = self.model.to('cpu')
                logger.info(f"Loaded fallback embedding model on {device}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise e2
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            else:
                self.embedding_cache = {}
                logger.info("No cache file found, starting with empty cache")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _get_cache_key(self, text: str, embedding_type: str = "general") -> str:
        """Generate cache key for text."""
        return f"{embedding_type}_{hash(text.lower().strip())}"
    
    def generate_embedding(self, text: str, embedding_type: str = "general", use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a text string."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(384)  # Default dimension for MiniLM
        
        text = text.strip()
        cache_key = self._get_cache_key(text, embedding_type)
        
        # Check cache first
        if use_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Generate new embedding
            embedding = self.model.encode(text)
            
            # Store in cache
            if use_cache:
                self.embedding_cache[cache_key] = embedding
                
                # Periodically save cache
                if len(self.embedding_cache) % 100 == 0:
                    self._save_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for '{text}': {e}")
            return np.zeros(384)
    
    def generate_merchant_embedding(self, merchant_name: str) -> np.ndarray:
        """Generate embedding specifically for merchant names."""
        # Preprocess merchant name for better embeddings
        merchant_clean = merchant_name.lower().strip()
        
        # Remove common noise
        import re
        merchant_clean = re.sub(r'[^\w\s]', ' ', merchant_clean)
        merchant_clean = re.sub(r'\s+', ' ', merchant_clean).strip()
        
        return self.generate_embedding(merchant_clean, "merchant")
    
    def generate_description_embedding(self, description: str) -> np.ndarray:
        """Generate embedding for transaction descriptions."""
        # Clean and preprocess description
        desc_clean = description.lower().strip()
        
        # Remove noise and normalize
        import re
        desc_clean = re.sub(r'[^\w\s]', ' ', desc_clean)
        desc_clean = re.sub(r'\s+', ' ', desc_clean).strip()
        
        return self.generate_embedding(desc_clean, "description")
    
    def generate_transaction_embeddings(self, transactions: List[TransactionData]) -> List[Dict[str, Any]]:
        """Generate embeddings for a batch of transactions."""
        results = []
        
        for transaction in transactions:
            try:
                # Generate merchant embedding
                merchant_embedding = self.generate_merchant_embedding(transaction.merchant_name)
                
                # Generate description embedding
                description_embedding = self.generate_description_embedding(transaction.description)
                
                # Combine embeddings (simple concatenation or averaging)
                combined_embedding = (merchant_embedding + description_embedding) / 2
                
                result = {
                    "transaction_id": transaction.id,
                    "merchant_embedding": merchant_embedding,
                    "description_embedding": description_embedding,
                    "combined_embedding": combined_embedding,
                    "embedding_metadata": {
                        "model": self.model_name,
                        "merchant": transaction.merchant_name,
                        "description": transaction.description
                    }
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for transaction {transaction.id}: {e}")
                continue
        
        # Save cache after batch processing
        self._save_cache()
        
        logger.info(f"Generated embeddings for {len(results)} transactions")
        return results
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_similar_merchants(self, target_merchant: str, merchant_embeddings: Dict[str, np.ndarray], 
                              top_k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find similar merchants based on embeddings."""
        try:
            target_embedding = self.generate_merchant_embedding(target_merchant)
            similarities = []
            
            for merchant, embedding in merchant_embeddings.items():
                if merchant.lower() == target_merchant.lower():
                    continue
                    
                similarity = self.compute_similarity(target_embedding, embedding)
                if similarity >= threshold:
                    similarities.append((merchant, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar merchants: {e}")
            return []
    
    def update_merchant_embeddings(self, new_merchants: List[str]) -> Dict[str, np.ndarray]:
        """Update embeddings for new merchants."""
        embeddings = {}
        
        for merchant in new_merchants:
            try:
                embedding = self.generate_merchant_embedding(merchant)
                embeddings[merchant] = embedding
                
            except Exception as e:
                logger.error(f"Error generating embedding for merchant '{merchant}': {e}")
                continue
        
        logger.info(f"Generated embeddings for {len(embeddings)} new merchants")
        return embeddings
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about cached embeddings."""
        merchant_count = len([k for k in self.embedding_cache.keys() if k.startswith("merchant_")])
        description_count = len([k for k in self.embedding_cache.keys() if k.startswith("description_")])
        general_count = len([k for k in self.embedding_cache.keys() if k.startswith("general_")])
        
        return {
            "total_cached": len(self.embedding_cache),
            "merchant_embeddings": merchant_count,
            "description_embeddings": description_count,
            "general_embeddings": general_count,
            "model_name": self.model_name,
            "cache_file": self.cache_file
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache = {}
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def run(self, transactions: List[TransactionData], **kwargs) -> Dict[str, Any]:
        """Agno agent run method."""
        try:
            # Generate embeddings for all transactions
            embeddings = self.generate_transaction_embeddings(transactions)
            
            # Get stats
            stats = self.get_embedding_stats()
            
            return {
                "status": "success",
                "embeddings_generated": len(embeddings),
                "embeddings": embeddings,
                "cache_stats": stats,
                "message": f"Generated embeddings for {len(embeddings)} transactions"
            }
            
        except Exception as e:
            logger.error(f"Embedding agent error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "embeddings": []
            }