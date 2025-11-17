"""
Retriever Agent - Handles similarity search and information retrieval in IntelliSpend
This agent manages the FAISS vector store and performs semantic searches for
similar merchants and transaction patterns.
"""

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import pickle
import os
import logging
from models.transaction import TransactionData
from utils.vector_store import MerchantVectorStore

logger = logging.getLogger(__name__)


class RetrieverAgent(Agent):
    """
    Agent responsible for similarity search and information retrieval.
    
    Key responsibilities:
    - Manage FAISS vector store
    - Perform similarity searches
    - Retrieve similar merchants
    - Find transaction patterns
    - Update vector indices
    """
    
    def __init__(self, vector_store_path: str = "data/vector_store"):
        super().__init__(
            name="RetrieverAgent", 
            description="Performs similarity search and pattern retrieval with advanced analysis tools",
            instructions=[
                "Maintain FAISS vector store for efficient similarity search",
                "Retrieve similar merchants for classification with pattern analysis",
                "Find transaction patterns and anomalies",
                "Search for merchant information when needed",
                "Update vector indices with new data and optimize performance",
                "Monitor vector store health and provide optimization suggestions"
            ],
            tools=[
                DuckDuckGoTools()  # Web search for merchant information
            ]
        )
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.merchant_index = None
        self.merchant_embeddings = {}
        self.merchant_categories = {}
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            # MerchantVectorStore() doesn't take a path parameter - it uses default settings
            self.vector_store = MerchantVectorStore()
            logger.info("Vector store initialized successfully")
            
            # Load existing data if available
            self._load_merchant_data()
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            # Create a basic fallback
            self.vector_store = MerchantVectorStore()
    
    def _load_merchant_data(self):
        """Load existing merchant embeddings and categories."""
        try:
            merchant_data_file = os.path.join(self.vector_store_path, "merchant_data.pkl")
            if os.path.exists(merchant_data_file):
                with open(merchant_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.merchant_embeddings = data.get('embeddings', {})
                    self.merchant_categories = data.get('categories', {})
                    
                logger.info(f"Loaded {len(self.merchant_embeddings)} merchant embeddings")
            else:
                # Initialize with sample data
                self._initialize_sample_data()
                
        except Exception as e:
            logger.error(f"Error loading merchant data: {e}")
            self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample merchant data."""
        self.merchant_categories = {
            'Amazon': 'Shopping',
            'Walmart': 'Shopping', 
            'Target': 'Shopping',
            'Starbucks': 'Food & Drink',
            'McDonald\'s': 'Food & Drink',
            'Shell': 'Gas',
            'Exxon': 'Gas',
            'Netflix': 'Entertainment',
            'Spotify': 'Entertainment',
            'Chase Bank': 'Banking',
            'Wells Fargo': 'Banking',
            'Uber': 'Transportation',
            'Lyft': 'Transportation'
        }
    
    def _save_merchant_data(self):
        """Save merchant embeddings and categories."""
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            merchant_data_file = os.path.join(self.vector_store_path, "merchant_data.pkl")
            
            data = {
                'embeddings': self.merchant_embeddings,
                'categories': self.merchant_categories
            }
            
            with open(merchant_data_file, 'wb') as f:
                pickle.dump(data, f)
                
            logger.debug("Merchant data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving merchant data: {e}")
    
    def add_merchant_embeddings(self, merchant_embeddings: Dict[str, np.ndarray], 
                               categories: Optional[Dict[str, str]] = None):
        """Add new merchant embeddings to the vector store."""
        try:
            # Update local storage
            self.merchant_embeddings.update(merchant_embeddings)
            
            if categories:
                self.merchant_categories.update(categories)
            
            # Add to vector store
            for merchant, embedding in merchant_embeddings.items():
                self.vector_store.add_merchant(merchant, "Unknown")  # Category will be updated later
            
            # Save to disk
            self._save_merchant_data()
            self.vector_store.save_index()
            
            logger.info(f"Added {len(merchant_embeddings)} merchant embeddings to vector store")
            
        except Exception as e:
            logger.error(f"Error adding merchant embeddings: {e}")
    
    def search_similar_merchants(self, merchant_name: str, 
                                top_k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float, str]]:
        """Search for similar merchants using the vector store."""
        try:
            if self.vector_store is None:
                logger.warning("Vector store not initialized")
                return []
            
            # Search using vector store - it expects merchant name, not embedding
            similar_merchants = self.vector_store.find_similar_merchants(
                merchant_name, k=top_k, min_similarity=threshold
            )
            
            # Filter by threshold and add categories - result is already filtered
            results = []
            for merchant_data, similarity in similar_merchants:
                # merchant_data is MerchantData object with name and category
                results.append((merchant_data.name, similarity, merchant_data.category))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar merchants: {e}")
            return []
    
    def find_merchant_by_name(self, merchant_name: str) -> Optional[Tuple[str, str]]:
        """Find exact or close merchant match by name."""
        try:
            # Use vector store to get category
            category = self.vector_store.get_category_for_merchant(merchant_name)
            if category and category != "Unknown":
                return merchant_name, category
            
            # Fallback to similarity search
            similar_merchants = self.search_similar_merchants(merchant_name, top_k=1, threshold=0.8)
            if similar_merchants:
                best_match = similar_merchants[0]
                return best_match[0], best_match[2]  # (name, category)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding merchant by name: {e}")
            return None
    
    def get_merchant_category(self, merchant_name: str) -> Optional[str]:
        """Get category for a known merchant."""
        try:
            return self.vector_store.get_category_for_merchant(merchant_name)
        except Exception:
            return self.merchant_categories.get(merchant_name)
    
    def search_transaction_patterns(self, transactions: List[TransactionData], 
                                   pattern_type: str = "spending") -> List[Dict[str, Any]]:
        """Search for transaction patterns."""
        try:
            patterns = []
            
            if pattern_type == "spending":
                # Find spending patterns by merchant
                merchant_spending = {}
                for transaction in transactions:
                    merchant = transaction.merchant
                    if merchant not in merchant_spending:
                        merchant_spending[merchant] = []
                    merchant_spending[merchant].append(transaction)
                
                # Analyze patterns
                for merchant, txns in merchant_spending.items():
                    if len(txns) >= 3:  # At least 3 transactions for pattern
                        total_amount = sum(t.amount for t in txns)
                        avg_amount = total_amount / len(txns)
                        
                        pattern = {
                            "type": "frequent_merchant",
                            "merchant": merchant,
                            "transaction_count": len(txns),
                            "total_amount": total_amount,
                            "average_amount": avg_amount,
                            "category": self.get_merchant_category(merchant),
                            "confidence": min(1.0, len(txns) / 10.0)  # Confidence based on frequency
                        }
                        patterns.append(pattern)
            
            elif pattern_type == "anomaly":
                # Find anomalous transactions
                if len(transactions) > 1:
                    amounts = [t.amount for t in transactions]
                    avg_amount = sum(amounts) / len(amounts)
                    std_amount = np.std(amounts)
                    
                    for transaction in transactions:
                        # Flag transactions more than 2 standard deviations from mean
                        if abs(transaction.amount - avg_amount) > 2 * std_amount:
                            pattern = {
                                "type": "amount_anomaly",
                                "transaction_id": transaction.id,
                                "merchant": transaction.merchant,
                                "amount": transaction.amount,
                                "expected_range": (avg_amount - std_amount, avg_amount + std_amount),
                                "confidence": min(1.0, abs(transaction.amount - avg_amount) / (3 * std_amount))
                            }
                            patterns.append(pattern)
            
            logger.info(f"Found {len(patterns)} patterns of type '{pattern_type}'")
            return patterns
            
        except Exception as e:
            logger.error(f"Error searching transaction patterns: {e}")
            return []
    
    def get_similar_transactions(self, target_transaction: TransactionData,
                                all_transactions: List[TransactionData],
                                similarity_threshold: float = 0.8) -> List[Tuple[TransactionData, float]]:
        """Find similar transactions based on merchant and amount."""
        try:
            similar = []
            
            for transaction in all_transactions:
                if transaction.id == target_transaction.id:
                    continue
                
                # Calculate similarity score
                similarity_score = 0.0
                
                # Merchant similarity (exact match gets full points)
                if transaction.merchant.lower() == target_transaction.merchant.lower():
                    similarity_score += 0.6
                elif target_transaction.merchant.lower() in transaction.merchant.lower():
                    similarity_score += 0.4
                elif transaction.merchant.lower() in target_transaction.merchant.lower():
                    similarity_score += 0.4
                
                # Amount similarity
                amount_diff = abs(transaction.amount - target_transaction.amount)
                max_amount = max(transaction.amount, target_transaction.amount)
                if max_amount > 0:
                    amount_similarity = 1 - (amount_diff / max_amount)
                    similarity_score += 0.4 * max(0, amount_similarity)
                
                if similarity_score >= similarity_threshold:
                    similar.append((transaction, similarity_score))
            
            # Sort by similarity
            similar.sort(key=lambda x: x[1], reverse=True)
            return similar
            
        except Exception as e:
            logger.error(f"Error finding similar transactions: {e}")
            return []
    
    def update_merchant_category(self, merchant: str, category: str):
        """Update merchant category mapping."""
        try:
            self.merchant_categories[merchant] = category
            self._save_merchant_data()
            logger.info(f"Updated category for '{merchant}' to '{category}'")
            
        except Exception as e:
            logger.error(f"Error updating merchant category: {e}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        try:
            vector_store_stats = self.vector_store.get_stats() if self.vector_store else {}
            
            return {
                "merchant_count": len(self.merchant_categories),
                "embeddings_count": len(self.merchant_embeddings),
                "categories": list(set(self.merchant_categories.values())),
                "vector_store_stats": vector_store_stats,
                "storage_path": self.vector_store_path
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {e}")
            return {}
    
    async def run(self, merchant_name: str = None, 
                  transactions: List[TransactionData] = None,
                  operation: str = "search", **kwargs) -> Dict[str, Any]:
        """Agno agent run method."""
        try:
            if operation == "search" and merchant_name:
                # Search for similar merchants using merchant name
                similar_merchants = self.search_similar_merchants(
                    merchant_name, 
                    top_k=kwargs.get('top_k', 5),
                    threshold=kwargs.get('threshold', 0.7)
                )
                
                return {
                    "status": "success",
                    "operation": "similarity_search",
                    "results": similar_merchants,
                    "message": f"Found {len(similar_merchants)} similar merchants"
                }
                
            elif operation == "lookup" and merchant_name:
                # Lookup merchant by name
                result = self.find_merchant_by_name(merchant_name)
                
                return {
                    "status": "success",
                    "operation": "merchant_lookup", 
                    "result": result,
                    "message": f"Lookup for merchant '{merchant_name}'"
                }
                
            elif operation == "patterns" and transactions:
                # Find transaction patterns
                patterns = self.search_transaction_patterns(
                    transactions,
                    pattern_type=kwargs.get('pattern_type', 'spending')
                )
                
                return {
                    "status": "success",
                    "operation": "pattern_search",
                    "patterns": patterns,
                    "message": f"Found {len(patterns)} patterns"
                }
            
            else:
                return {
                    "status": "error",
                    "error": "Invalid operation or missing parameters"
                }
                
        except Exception as e:
            logger.error(f"Retriever agent error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }