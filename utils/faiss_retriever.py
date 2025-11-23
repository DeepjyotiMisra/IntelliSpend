"""FAISS retriever for merchant similarity search"""

import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from utils.embedding_service import get_embedding_service
from config.settings import settings

logger = logging.getLogger(__name__)

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FAISSRetriever:
    """
    RAG-style retriever for merchant similarity search.
    Uses FAISS index with normalized embeddings for cosine similarity.
    """
    
    def __init__(self,
                 index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 embedding_service=None,
                 top_k: int = 5):
        """
        Initialize FAISS retriever.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            embedding_service: EmbeddingService instance (if None, creates one)
            top_k: Default number of results to retrieve
        """
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.metadata_path = Path(metadata_path or settings.MERCHANT_METADATA_PATH)
        self.top_k = top_k
        self.embedding_service = embedding_service or get_embedding_service()
        
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._load_time: float = 0.0  # Track when index was loaded
        
        # Load index if it exists
        if self.index_path.exists() and self.metadata_path.exists():
            self._load_index()
        else:
            logger.warning(
                f"FAISS index not found at {self.index_path}. "
                "Please build the index first using faiss_index_builder."
            )
    
    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
        
        import time
        start_time = time.time()
        
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))
        index_load_time = time.time() - start_time
        
        logger.info(f"Loading metadata from {self.metadata_path}...")
        metadata_start = time.time()
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        metadata_load_time = time.time() - metadata_start
        
        # Validate metadata structure
        if not isinstance(self.metadata, list):
            raise ValueError("Metadata should be a list of dicts")
        
        # Store load time for comparison with file modification time
        self._load_time = self.index_path.stat().st_mtime
        
        total_load_time = time.time() - start_time
        logger.info(
            f"Loaded index with {self.index.ntotal} vectors and "
            f"{len(self.metadata)} metadata entries "
            f"(index: {index_load_time:.2f}s, metadata: {metadata_load_time:.2f}s, total: {total_load_time:.2f}s)"
        )
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single string or list of strings
            batch_size: Batch size for embedding generation (default: uses settings.BATCH_SIZE)
            
        Returns:
            Normalized embeddings as numpy array
        """
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
        return self.embedding_service.encode(
            texts,
            normalize=True,  # Normalize for cosine similarity
            batch_size=batch_size,
            show_progress_bar=False
        )
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar merchants for a query.
        
        Args:
            query: Transaction description to search for
            top_k: Number of results to return (default: self.top_k)
            min_score: Minimum similarity score threshold (optional)
            
        Returns:
            List of dicts with keys: id, merchant, description, category, score
        """
        if self.index is None:
            raise ValueError("Index not loaded. Please build or load the index first.")
        
        if top_k is None:
            top_k = self.top_k
        
        # Generate query embedding
        query_embedding = self.embed(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for score, idx in zip(distances[0], indices[0]):
            # Skip invalid indices
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            # Apply minimum score threshold if specified
            if min_score is not None and score < min_score:
                continue
            
            meta = self.metadata[idx]
            results.append({
                'id': meta.get('id', idx),
                'merchant': meta.get('merchant', 'Unknown'),
                'description': meta.get('description', ''),
                'description_normalized': meta.get('description_normalized', ''),
                'category': meta.get('category', 'Unknown'),
                'score': float(score)  # Cosine similarity score (0-1)
            })
        
        return results
    
    def batch_retrieve(self,
                      queries: List[str],
                      top_k: Optional[int] = None,
                      min_score: Optional[float] = None) -> List[List[Dict[str, Any]]]:
        """
        Retrieve similar merchants for multiple queries (batch processing).
        
        Args:
            queries: List of transaction descriptions
            top_k: Number of results per query
            min_score: Minimum similarity score threshold
            
        Returns:
            List of result lists (one per query)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Please build or load the index first.")
        
        if top_k is None:
            top_k = self.top_k
        
        if not queries:
            return []
        
        # Generate embeddings for all queries
        query_embeddings = self.embed(queries)
        
        # Batch search in FAISS index
        distances, indices = self.index.search(query_embeddings, top_k)
        
        # Build results for each query
        all_results = []
        for query_idx, (row_scores, row_indices) in enumerate(zip(distances, indices)):
            query_results = []
            for score, idx in zip(row_scores, row_indices):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                # Apply minimum score threshold if specified
                if min_score is not None and score < min_score:
                    continue
                
                meta = self.metadata[idx]
                query_results.append({
                    'id': meta.get('id', idx),
                    'merchant': meta.get('merchant', 'Unknown'),
                    'description': meta.get('description', ''),
                    'description_normalized': meta.get('description_normalized', ''),
                    'category': meta.get('category', 'Unknown'),
                    'score': float(score)
                })
            all_results.append(query_results)
        
        return all_results
    
    def get_best_match(self,
                      query: str,
                      min_score: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get the best matching merchant for a query.
        
        Args:
            query: Transaction description
            min_score: Minimum similarity score threshold
            
        Returns:
            Best match dict or None if no match above threshold
        """
        results = self.retrieve(query, top_k=1, min_score=min_score)
        return results[0] if results else None
    
    def explain(self,
               query: str,
               top_k: Optional[int] = None) -> str:
        """
        Generate a human-readable explanation of retrieval results.
        
        Args:
            query: Transaction description
            top_k: Number of results to include
            
        Returns:
            Formatted explanation string
        """
        matches = self.retrieve(query, top_k=top_k)
        
        if not matches:
            return f"No matches found for: '{query}'"
        
        lines = [f"Top {len(matches)} matches for: '{query}'"]
        lines.append("=" * 60)
        
        for i, match in enumerate(matches, 1):
            lines.append(
                f"{i}. {match['merchant']} "
                f"(score={match['score']:.3f}, category={match['category']})"
            )
            lines.append(f"   Description: '{match['description']}'")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_category_from_match(self,
                               query: str,
                               min_score: Optional[float] = None) -> Optional[str]:
        """
        Get category from best matching merchant.
        
        Args:
            query: Transaction description
            min_score: Minimum similarity score threshold
            
        Returns:
            Category name or None
        """
        best_match = self.get_best_match(query, min_score=min_score)
        return best_match['category'] if best_match else None
    
    def is_index_loaded(self) -> bool:
        """Check if index is loaded"""
        return self.index is not None
    
    def reload_index(self) -> None:
        """
        Reload the FAISS index and metadata from disk.
        Use this after the index has been rebuilt to ensure you're using the latest version.
        """
        logger.info("Reloading FAISS index from disk...")
        self._load_index()
        logger.info("Index reloaded successfully")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded index"""
        if self.index is None:
            return {'status': 'not_loaded'}
        
        stats = {
            'num_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'index_type': type(self.index).__name__,
            'metadata_count': len(self.metadata)
        }
        
        return stats


# Global singleton instance (lazy loaded)
_retriever: Optional[FAISSRetriever] = None


def get_retriever(index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 top_k: int = 5,
                 force_reload: bool = False) -> FAISSRetriever:
    """
    Get or create the global retriever instance.
    
    Args:
        index_path: Optional index path override
        metadata_path: Optional metadata path override
        top_k: Default number of results
        force_reload: If True, reload the retriever even if it exists (useful after index rebuild)
        
    Returns:
        FAISSRetriever instance
    """
    global _retriever
    
    # Check if we need to reload due to index file modification
    if _retriever is not None and not force_reload:
        # Check if index file was modified after retriever was loaded
        actual_index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        if actual_index_path.exists():
            try:
                index_mtime = actual_index_path.stat().st_mtime
                # If retriever was loaded before index was modified, reload it
                retriever_load_time = getattr(_retriever, '_load_time', 0)
                if not hasattr(_retriever, '_load_time') or retriever_load_time < index_mtime:
                    logger.info(f"Index file modified after retriever load (index_mtime={index_mtime}, load_time={retriever_load_time}), reloading retriever...")
                    force_reload = True
            except (OSError, AttributeError) as e:
                logger.warning(f"Could not check index modification time: {e}, forcing reload")
                force_reload = True
    
    if _retriever is None or force_reload:
        _retriever = FAISSRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            top_k=top_k
        )
        # Store load time for comparison
        if _retriever.index_path.exists():
            _retriever._load_time = _retriever.index_path.stat().st_mtime
        else:
            _retriever._load_time = 0
    
    return _retriever


def reload_retriever(index_path: Optional[str] = None,
                    metadata_path: Optional[str] = None,
                    top_k: int = 5) -> FAISSRetriever:
    """
    Reload the global retriever instance (clears cache and reloads from disk).
    Use this after rebuilding the FAISS index to ensure the retriever uses the updated index.
    
    Args:
        index_path: Optional index path override
        metadata_path: Optional metadata path override
        top_k: Default number of results
        
    Returns:
        Newly loaded FAISSRetriever instance
    """
    global _retriever
    
    logger.info("Reloading FAISS retriever (clearing cache and reloading index from disk)...")
    _retriever = None  # Clear the cached instance
    
    # Create new instance which will load the updated index
    _retriever = FAISSRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        top_k=top_k
    )
    
    # Store load time
    if _retriever.index_path.exists():
        _retriever._load_time = _retriever.index_path.stat().st_mtime
    else:
        _retriever._load_time = 0
    
    logger.info("FAISS retriever reloaded successfully")
    return _retriever


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test retriever
    print("🔍 Testing FAISS Retriever...\n")
    
    retriever = get_retriever()
    
    if not retriever.is_index_loaded():
        print("❌ Index not loaded. Please build the index first:")
        print("   python utils/faiss_index_builder.py")
        exit(1)
    
    # Print index stats
    stats = retriever.get_index_stats()
    print("📊 Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test queries
    test_queries = [
        "AMAZON PAY INDIA TXN 12345",
        "UBER TRIP MUMBAI",
        "STARBUCKS COFFEE",
        "SHELL PETROL PUMP"
    ]
    
    for query in test_queries:
        print(retriever.explain(query, top_k=3))
        print()

