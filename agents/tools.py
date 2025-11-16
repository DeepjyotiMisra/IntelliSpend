"""Custom tools for IntelliSpend agents"""

from typing import List, Dict, Any, Optional
import logging

from utils.data_utils import normalize_transaction_string, extract_payment_mode
from utils.faiss_retriever import get_retriever

logger = logging.getLogger(__name__)


def normalize_transaction(description: str) -> Dict[str, Any]:
    """
    Normalize a transaction description string.
    
    Args:
        description: Raw transaction description
        
    Returns:
        Dict with normalized description and extracted metadata
    """
    try:
        normalized = normalize_transaction_string(description)
        payment_mode = extract_payment_mode(description)
        
        return {
            "success": True,
            "original": description,
            "normalized": normalized,
            "payment_mode": payment_mode,
            "length": len(normalized)
        }
    except Exception as e:
        logger.error(f"Error normalizing transaction: {e}")
        return {
            "success": False,
            "error": str(e),
            "original": description
        }


def retrieve_similar_merchants(description: str, top_k: int = 5, min_score: Optional[float] = None) -> Dict[str, Any]:
    """
    Retrieve similar merchants for a transaction description.
    
    Args:
        description: Transaction description (will be normalized automatically)
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        
    Returns:
        Dict with retrieval results
    """
    try:
        # Normalize the description first
        normalized = normalize_transaction_string(description)
        
        # Get retriever
        retriever = get_retriever(top_k=top_k)
        
        if not retriever.is_index_loaded():
            return {
                "success": False,
                "error": "FAISS index not loaded. Please build the index first.",
                "description": description
            }
        
        # Retrieve similar merchants
        results = retriever.retrieve(normalized, top_k=top_k, min_score=min_score)
        
        return {
            "success": True,
            "query": description,
            "normalized_query": normalized,
            "num_results": len(results),
            "results": results,
            "best_match": results[0] if results else None
        }
    except Exception as e:
        logger.error(f"Error retrieving merchants: {e}")
        return {
            "success": False,
            "error": str(e),
            "description": description
        }


def batch_normalize_transactions(descriptions: List[str]) -> Dict[str, Any]:
    """
    Normalize multiple transaction descriptions in batch.
    
    Args:
        descriptions: List of transaction descriptions
        
    Returns:
        Dict with batch normalization results
    """
    try:
        normalized_list = []
        payment_modes = []
        
        for desc in descriptions:
            normalized = normalize_transaction_string(desc)
            payment_mode = extract_payment_mode(desc)
            normalized_list.append(normalized)
            payment_modes.append(payment_mode)
        
        return {
            "success": True,
            "count": len(descriptions),
            "normalized": normalized_list,
            "payment_modes": payment_modes
        }
    except Exception as e:
        logger.error(f"Error batch normalizing transactions: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def batch_retrieve_merchants(descriptions: List[str], top_k: int = 5) -> Dict[str, Any]:
    """
    Retrieve similar merchants for multiple transaction descriptions in batch.
    
    Args:
        descriptions: List of transaction descriptions
        top_k: Number of results per query
        
    Returns:
        Dict with batch retrieval results
    """
    try:
        # Normalize all descriptions
        normalized = [normalize_transaction_string(desc) for desc in descriptions]
        
        # Get retriever
        retriever = get_retriever(top_k=top_k)
        
        if not retriever.is_index_loaded():
            return {
                "success": False,
                "error": "FAISS index not loaded. Please build the index first."
            }
        
        # Batch retrieve
        all_results = retriever.batch_retrieve(normalized, top_k=top_k)
        
        # Format results
        formatted_results = []
        for i, (desc, results) in enumerate(zip(descriptions, all_results)):
            formatted_results.append({
                "index": i,
                "original": desc,
                "normalized": normalized[i],
                "num_matches": len(results),
                "matches": results,
                "best_match": results[0] if results else None
            })
        
        return {
            "success": True,
            "count": len(descriptions),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error batch retrieving merchants: {e}")
        return {
            "success": False,
            "error": str(e)
        }

