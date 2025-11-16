"""Embedding service for generating vector embeddings using sentence-transformers"""

import os
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EmbeddingService:
    """
    Service for generating embeddings using sentence-transformers.
    Handles model loading, caching, and batch processing.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       If None, uses default from settings.
        """
        from config.settings import settings
        
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None
        
        logger.info(f"Initializing EmbeddingService with model: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model (lazy loading)"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Get embedding dimension by encoding a test string
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                self._embedding_dim = test_embedding.shape[1]
                
                logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dim}")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim
    
    def encode(self, 
               texts: Union[str, List[str]], 
               normalize: bool = True,
               batch_size: int = 32,
               show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single string or list of strings to encode
            normalize: Whether to L2-normalize embeddings (default: True for cosine similarity)
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if self.model is None:
            self._load_model()
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize
            )
            
            # Ensure it's float32 for FAISS compatibility
            embeddings = embeddings.astype('float32')
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_batch(self,
                    texts: List[str],
                    normalize: bool = True,
                    batch_size: int = 128,
                    show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode a batch of texts (optimized for large batches).
        
        Args:
            texts: List of strings to encode
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding (larger for batch processing)
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        return self.encode(
            texts,
            normalize=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length (L2 normalization).
        
        Args:
            embeddings: numpy array of embeddings
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9  # Avoid division by zero
        return (embeddings / norms).astype('float32')


# Global singleton instance (lazy loaded)
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=model_name)
    elif model_name and _embedding_service.model_name != model_name:
        # Recreate if model name changed
        _embedding_service = EmbeddingService(model_name=model_name)
    
    return _embedding_service

