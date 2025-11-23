"""FAISS index builder for merchant embeddings"""

import os
import sys
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.embedding_service import get_embedding_service
from utils.data_utils import load_merchant_seed_data
from config.settings import settings

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """
    Builds and manages FAISS index for merchant similarity search.
    Uses inner product (cosine similarity) on normalized embeddings.
    """
    
    def __init__(self, 
                 embedding_service=None,
                 index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """
        Initialize FAISS index builder.
        
        Args:
            embedding_service: EmbeddingService instance (if None, creates one)
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.metadata_path = Path(metadata_path or settings.MERCHANT_METADATA_PATH)
        
        # Ensure parent directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
    
    def build_index_from_seed_data(self, 
                                   seed_file: Optional[str] = None,
                                   use_normalized: bool = True) -> None:
        """
        Build FAISS index from merchant seed data.
        
        Args:
            seed_file: Path to merchants_seed.csv (if None, uses default from settings)
            use_normalized: Whether to use normalized descriptions
        """
        logger.info("Building FAISS index from merchant seed data...")
        
        # Load merchant seed data
        df = load_merchant_seed_data(seed_file)
        
        if df.empty:
            raise ValueError("Merchant seed data is empty")
        
        # Use normalized descriptions if available, otherwise use original
        description_col = 'description_normalized' if use_normalized and 'description_normalized' in df.columns else 'description'
        descriptions = df[description_col].tolist()
        
        logger.info(f"Processing {len(descriptions)} merchant descriptions...")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_service.encode_batch(
            descriptions,
            normalize=True,  # Normalize for cosine similarity
            batch_size=settings.BATCH_SIZE,
            show_progress_bar=True
        )
        
        # Create metadata list
        self.metadata = []
        for idx, row in df.iterrows():
            self.metadata.append({
                'id': int(idx),
                'merchant': str(row['merchant']),
                'description': str(row['description']),
                'description_normalized': str(row.get('description_normalized', row['description'])),
                'category': str(row.get('category', 'Unknown')),
                'sample_amount': float(row.get('sample_amount', 0.0)) if pd.notna(row.get('sample_amount')) else 0.0
            })
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index with dimension {dimension}...")
        
        # Use IndexFlatIP (Inner Product) for cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        logger.info(f"Index built successfully with {self.index.ntotal} vectors")
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        if not self.metadata:
            raise ValueError("No metadata to save. Build index first.")
        
        logger.info(f"Saving FAISS index to {self.index_path}...")
        faiss.write_index(self.index, str(self.index_path))
        
        logger.info(f"Saving metadata to {self.metadata_path}...")
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info("Index and metadata saved successfully")
    
    def load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
        
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Validate metadata structure
        if not isinstance(self.metadata, list):
            raise ValueError("Metadata should be a list of dicts")
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
    
    def add_merchants(self, 
                     merchants: List[Dict[str, Any]],
                     descriptions: List[str]) -> None:
        """
        Add new merchants to existing index.
        
        Args:
            merchants: List of merchant dicts with keys: merchant, category, etc.
            descriptions: List of merchant descriptions
        """
        if self.index is None:
            raise ValueError("No index loaded. Load or build index first.")
        
        if len(merchants) != len(descriptions):
            raise ValueError("Number of merchants and descriptions must match")
        
        logger.info(f"Adding {len(merchants)} new merchants to index...")
        
        # Generate embeddings for new descriptions
        embeddings = self.embedding_service.encode_batch(
            descriptions,
            normalize=True,
            batch_size=settings.BATCH_SIZE,
            show_progress_bar=False
        )
        
        # Add to index
        start_id = len(self.metadata)
        self.index.add(embeddings)
        
        # Add metadata
        for idx, merchant in enumerate(merchants):
            self.metadata.append({
                'id': start_id + idx,
                'merchant': str(merchant.get('merchant', 'Unknown')),
                'description': str(descriptions[idx]),
                'description_normalized': str(descriptions[idx]),
                'category': str(merchant.get('category', 'Unknown')),
                'sample_amount': float(merchant.get('sample_amount', 0.0))
            })
        
        logger.info(f"Added {len(merchants)} merchants. Total vectors: {self.index.ntotal}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if self.index is None:
            return {'status': 'no_index'}
        
        stats = {
            'num_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'index_type': type(self.index).__name__,
            'metadata_count': len(self.metadata)
        }
        
        # Count merchants by category
        if self.metadata:
            category_counts = {}
            for meta in self.metadata:
                category = meta.get('category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            stats['category_distribution'] = category_counts
        
        return stats


def build_merchant_index(seed_file: Optional[str] = None,
                        index_path: Optional[str] = None,
                        metadata_path: Optional[str] = None,
                        save: bool = True) -> FAISSIndexBuilder:
    """
    Convenience function to build merchant index from seed data.
    
    Args:
        seed_file: Path to merchants_seed.csv
        index_path: Path to save FAISS index
        metadata_path: Path to save metadata
        save: Whether to save index to disk
        
    Returns:
        FAISSIndexBuilder instance with built index
    """
    builder = FAISSIndexBuilder(
        index_path=index_path,
        metadata_path=metadata_path
    )
    
    builder.build_index_from_seed_data(seed_file=seed_file)
    
    if save:
        builder.save_index()
    
    return builder


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Build index
    print("🚀 Building merchant FAISS index...")
    builder = build_merchant_index()
    
    # Print statistics
    stats = builder.get_index_stats()
    print("\n📊 Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Index build complete!")

