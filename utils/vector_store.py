"""
FAISS Vector Store for Merchant Similarity Search

Implements vector-based merchant matching using FAISS for fast similarity search.
Enables finding similar merchants to improve transaction categorization accuracy.
"""

import os
import pickle
import logging
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from pathlib import Path

from models.transaction import TransactionData, CategorizedTransaction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MerchantData:
    """Merchant information with categorization history"""
    name: str
    normalized_name: str
    category: str
    subcategory: Optional[str]
    transaction_count: int = 0
    confidence_score: float = 0.0
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class MerchantVectorStore:
    """FAISS-based vector store for merchant similarity search"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 data_dir: str = "data/vectors"):
        """
        Initialize the merchant vector store
        
        Args:
            model_name: SentenceTransformer model for embeddings
            data_dir: Directory to store FAISS indices and metadata
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentence transformer with CPU device
        logger.info(f"Loading embedding model: {model_name}")
        import torch
        import os
        
        # Set environment variables to force CPU usage
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        device = 'cpu'  # Force CPU to avoid MPS tensor issues
        self.encoder = SentenceTransformer(model_name, device=device)
        
        # Ensure model is on CPU
        self.encoder = self.encoder.to('cpu')
        
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # FAISS index and metadata
        self.index = None
        self.merchant_data: List[MerchantData] = []
        self.merchant_lookup: Dict[str, int] = {}  # merchant_name -> index
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self) -> None:
        """Load existing FAISS index and merchant data"""
        index_path = self.data_dir / "merchant_index.faiss"
        metadata_path = self.data_dir / "merchant_metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                logger.info("Loading existing merchant vector store...")
                self.index = faiss.read_index(str(index_path))
                
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.merchant_data = data['merchants']
                    self.merchant_lookup = data['lookup']
                
                logger.info(f"Loaded {len(self.merchant_data)} merchants from existing store")
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
                self._initialize_empty_index()
        else:
            self._initialize_empty_index()
    
    def _initialize_empty_index(self) -> None:
        """Initialize empty FAISS index"""
        logger.info("Initializing new merchant vector store...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
        self.merchant_data = []
        self.merchant_lookup = {}
    
    def add_merchant(self, 
                    name: str, 
                    category: str,
                    subcategory: Optional[str] = None,
                    aliases: Optional[List[str]] = None,
                    confidence_score: float = 1.0) -> None:
        """
        Add a new merchant to the vector store
        
        Args:
            name: Merchant name
            category: Primary category
            subcategory: Optional subcategory
            aliases: Alternative names for the merchant
            confidence_score: Confidence in this categorization
        """
        normalized_name = self._normalize_merchant_name(name)
        
        # Check if merchant already exists
        if normalized_name in self.merchant_lookup:
            logger.info(f"Merchant already exists: {normalized_name}")
            return
        
        # Create merchant data
        merchant = MerchantData(
            name=name,
            normalized_name=normalized_name,
            category=category,
            subcategory=subcategory,
            transaction_count=1,
            confidence_score=confidence_score,
            aliases=aliases or []
        )
        
        # Generate embedding
        embedding_text = self._create_embedding_text(merchant)
        embedding = self.encoder.encode([embedding_text])
        
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embedding.astype(np.float32))
        
        # Add to metadata
        merchant_idx = len(self.merchant_data)
        self.merchant_data.append(merchant)
        self.merchant_lookup[normalized_name] = merchant_idx
        
        logger.info(f"Added merchant: {name} -> {category}")
    
    def find_similar_merchants(self, 
                             merchant_name: str, 
                             k: int = 5,
                             min_similarity: float = 0.7) -> List[Tuple[MerchantData, float]]:
        """
        Find similar merchants based on name similarity
        
        Args:
            merchant_name: Name to search for
            k: Number of similar merchants to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (MerchantData, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.encoder.encode([merchant_name])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search FAISS index
        similarities, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Filter results by minimum similarity
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= min_similarity and idx < len(self.merchant_data):
                merchant = self.merchant_data[idx]
                results.append((merchant, float(similarity)))
        
        return results
    
    def update_merchant_stats(self, merchant_name: str, category: str) -> None:
        """Update merchant statistics based on successful categorization"""
        normalized_name = self._normalize_merchant_name(merchant_name)
        
        if normalized_name in self.merchant_lookup:
            idx = self.merchant_lookup[normalized_name]
            merchant = self.merchant_data[idx]
            merchant.transaction_count += 1
            
            # Update confidence if category matches
            if merchant.category == category:
                merchant.confidence_score = min(1.0, merchant.confidence_score + 0.1)
        else:
            # Add as new merchant
            self.add_merchant(merchant_name, category)
    
    def get_category_for_merchant(self, merchant_name: str) -> Optional[Tuple[str, str, float]]:
        """
        Get stored category for a known merchant
        
        Returns:
            Tuple of (category, subcategory, confidence) or None if not found
        """
        normalized_name = self._normalize_merchant_name(merchant_name)
        
        if normalized_name in self.merchant_lookup:
            idx = self.merchant_lookup[normalized_name]
            merchant = self.merchant_data[idx]
            return (merchant.category, merchant.subcategory, merchant.confidence_score)
        
        return None
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            index_path = self.data_dir / "merchant_index.faiss"
            metadata_path = self.data_dir / "merchant_metadata.pkl"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'merchants': self.merchant_data,
                'lookup': self.merchant_lookup
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved merchant vector store with {len(self.merchant_data)} merchants")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _normalize_merchant_name(self, name: str) -> str:
        """Normalize merchant name for consistent lookup"""
        return name.strip().upper().replace('*', '').replace('#', '')
    
    def _create_embedding_text(self, merchant: MerchantData) -> str:
        """Create text for embedding generation"""
        text_parts = [merchant.normalized_name]
        
        # Add aliases
        if merchant.aliases:
            text_parts.extend(merchant.aliases)
        
        # Add category context
        text_parts.append(merchant.category)
        if merchant.subcategory:
            text_parts.append(merchant.subcategory)
        
        return " ".join(text_parts)
    
    def load_from_transactions(self, transactions: List[CategorizedTransaction]) -> None:
        """
        Load merchants from a list of categorized transactions
        
        Args:
            transactions: List of categorized transactions to learn from
        """
        logger.info(f"Learning from {len(transactions)} transactions...")
        
        for transaction in transactions:
            if transaction.is_verified and transaction.final_category:
                self.add_merchant(
                    name=transaction.transaction.merchant_name,
                    category=transaction.final_category,
                    subcategory=transaction.primary_prediction.subcategory,
                    confidence_score=transaction.primary_prediction.confidence_score
                )
        
        # Save after loading
        self.save_index()
        logger.info("Merchant learning complete!")


def initialize_sample_merchants(vector_store: MerchantVectorStore) -> None:
    """Initialize the vector store with sample merchants for testing"""
    
    sample_merchants = [
        # Food & Dining
        ("McDonald's", "food_dining", "fast_food"),
        ("Starbucks", "food_dining", "coffee_shops"),
        ("Subway", "food_dining", "fast_food"),
        ("Whole Foods", "food_dining", "groceries"),
        ("Safeway", "food_dining", "groceries"),
        
        # Transportation
        ("Shell", "transportation", "gas_fuel"),
        ("BP", "transportation", "gas_fuel"),
        ("Uber", "transportation", "rideshare"),
        ("Lyft", "transportation", "rideshare"),
        
        # Shopping
        ("Amazon", "shopping", "general_merchandise"),
        ("Target", "shopping", "general_merchandise"),
        ("Best Buy", "shopping", "electronics"),
        ("Apple Store", "shopping", "electronics"),
        
        # Utilities
        ("Pacific Gas & Electric", "utilities_bills", "gas"),
        ("Comcast", "utilities_bills", "internet"),
        ("AT&T", "utilities_bills", "phone"),
        
        # Entertainment
        ("Netflix", "entertainment", "streaming_services"),
        ("Spotify", "entertainment", "streaming_services"),
        ("AMC Theaters", "entertainment", "movies_theater"),
    ]
    
    for merchant_name, category, subcategory in sample_merchants:
        vector_store.add_merchant(merchant_name, category, subcategory)
    
    vector_store.save_index()
    logger.info(f"Initialized vector store with {len(sample_merchants)} sample merchants")


if __name__ == "__main__":
    # Test the merchant vector store
    vector_store = MerchantVectorStore()
    
    # Initialize with sample data if empty
    if len(vector_store.merchant_data) == 0:
        initialize_sample_merchants(vector_store)
    
    # Test similarity search
    test_queries = ["McDonalds", "Star Bucks", "Amazon.com", "Shell Gas"]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        similar = vector_store.find_similar_merchants(query, k=3)
        for merchant, score in similar:
            print(f"  {merchant.name} ({merchant.category}) - {score:.3f}")