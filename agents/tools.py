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


def get_taxonomy_categories() -> Dict[str, Any]:
    """
    Get available categories from taxonomy configuration, including custom categories.
    
    Returns:
        Dict with taxonomy information including categories and their descriptions
    """
    try:
        from config.settings import settings
        
        taxonomy = settings.load_taxonomy()
        standard_categories = settings.get_categories()
        standard_category_names = settings.get_category_names()
        
        # Include custom categories
        try:
            from utils.custom_categories import get_all_custom_categories, get_all_categories_with_custom
            custom_categories = get_all_custom_categories()
            all_category_names = get_all_categories_with_custom()
            
            # Combine standard and custom categories
            all_categories = standard_categories + custom_categories
        except Exception as e:
            logger.warning(f"Could not load custom categories: {e}")
            custom_categories = []
            all_categories = standard_categories
            all_category_names = standard_category_names
        
        return {
            "success": True,
            "categories": all_categories,
            "category_names": all_category_names,
            "standard_categories": standard_categories,
            "custom_categories": custom_categories,
            "version": taxonomy.get("version", "1.0")
        }
    except Exception as e:
        logger.error(f"Error loading taxonomy: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def store_feedback(
    transaction_description: str,
    original_merchant: str,
    original_category: str,
    corrected_merchant: str,
    corrected_category: str,
    amount: float = None,
    date: str = None,
    confidence_score: float = None,
    feedback_type: str = "correction"
) -> Dict[str, Any]:
    """
    Store user feedback/correction for a transaction classification.
    
    Args:
        transaction_description: Original transaction description
        original_merchant: Merchant that was originally classified
        original_category: Category that was originally assigned
        corrected_merchant: Correct merchant name (user correction)
        corrected_category: Correct category (user correction)
        amount: Transaction amount (optional)
        date: Transaction date (optional)
        confidence_score: Original confidence score (optional)
        feedback_type: Type of feedback - "correction", "new_merchant", "category_fix" (default: "correction")
        
    Returns:
        Dict with feedback storage result
    """
    try:
        import pandas as pd
        from pathlib import Path
        from config.settings import settings
        from datetime import datetime
        
        feedback_path = Path(settings.FEEDBACK_STORAGE_PATH)
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create feedback record
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'transaction_description': transaction_description,
            'original_merchant': original_merchant,
            'original_category': original_category,
            'corrected_merchant': corrected_merchant,
            'corrected_category': corrected_category,
            'amount': amount,
            'date': date,
            'confidence_score': confidence_score,
            'feedback_type': feedback_type,
            'processed': False  # Flag to track if feedback has been applied
        }
        
        # Load existing feedback or create new DataFrame
        if feedback_path.exists():
            df = pd.read_csv(feedback_path)
        else:
            df = pd.DataFrame(columns=feedback_record.keys())
        
        # Check for duplicates before adding
        # Consider duplicate if same: transaction_description, corrected_category, amount, date
        # Note: We check corrected_category (the correction) rather than corrected_merchant
        # because the same transaction correction might have different merchant values if submitted multiple times
        key_columns = ['transaction_description', 'corrected_category']
        if amount is not None:
            key_columns.append('amount')
        if date:
            key_columns.append('date')
        
        # Filter to available columns
        available_key_cols = [col for col in key_columns if col in df.columns]
        
        if available_key_cols and len(df) > 0:
            # Check if this feedback already exists
            mask = pd.Series([True] * len(df), index=df.index)
            
            for col in available_key_cols:
                if col == 'transaction_description':
                    record_val = feedback_record.get(col, '')
                    if pd.isna(record_val) or str(record_val).lower() == 'nan' or record_val == '':
                        mask = mask & (df[col].isna() | (df[col].astype(str).str.lower() == 'nan') | (df[col] == ''))
                    else:
                        mask = mask & (df[col] == record_val)
                elif col == 'amount':
                    record_val = feedback_record.get(col)
                    if pd.isna(record_val):
                        mask = mask & df[col].isna()
                    else:
                        # Match amount considering absolute value (same transaction, different sign = duplicate)
                        # Also match exact value
                        mask = mask & ((df[col] == record_val) | (df[col].abs() == abs(record_val)))
                elif col == 'date':
                    record_val = feedback_record.get(col, '')
                    if pd.isna(record_val) or str(record_val).lower() == 'nan' or record_val == '':
                        mask = mask & (df[col].isna() | (df[col].astype(str).str.lower() == 'nan') | (df[col] == ''))
                    else:
                        mask = mask & (df[col] == record_val)
                else:
                    mask = mask & (df[col] == feedback_record.get(col, ''))
            
            # Also check if not already processed (only check pending feedback)
            if 'processed' in df.columns:
                mask = mask & (df['processed'] == False)
            
            if mask.any():
                # Duplicate found - don't add
                logger.info(f"Duplicate feedback detected, skipping: {corrected_merchant} / {corrected_category}")
                return {
                    "success": True,
                    "message": "Feedback already exists (duplicate skipped)",
                    "duplicate": True,
                    "feedback_id": df[mask].index[0]
                }
        
        # Append new feedback (no duplicate found)
        new_row = pd.DataFrame([feedback_record])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(feedback_path, index=False)
        
        logger.info(f"Feedback stored: {corrected_merchant} / {corrected_category}")
        
        return {
            "success": True,
            "message": "Feedback stored successfully",
            "feedback_id": len(df) - 1,
            "feedback_type": feedback_type
        }
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_feedback_statistics() -> Dict[str, Any]:
    """
    Get statistics about collected feedback.
    
    Returns:
        Dict with feedback statistics
    """
    try:
        import pandas as pd
        from pathlib import Path
        from config.settings import settings
        
        feedback_path = Path(settings.FEEDBACK_STORAGE_PATH)
        
        if not feedback_path.exists():
            return {
                "success": True,
                "total_feedback": 0,
                "processed": 0,
                "pending": 0,
                "most_corrected_merchants": [],
                "most_corrected_categories": []
            }
        
        df = pd.read_csv(feedback_path)
        
        # Calculate statistics
        total = len(df)
        processed = df['processed'].sum() if 'processed' in df.columns else 0
        pending = total - processed
        
        # Most corrected merchants
        if 'corrected_merchant' in df.columns:
            merchant_counts = df['corrected_merchant'].value_counts().head(10).to_dict()
        else:
            merchant_counts = {}
        
        # Most corrected categories
        if 'corrected_category' in df.columns:
            category_counts = df['corrected_category'].value_counts().head(10).to_dict()
        else:
            category_counts = {}
        
        return {
            "success": True,
            "total_feedback": total,
            "processed": int(processed),
            "pending": int(pending),
            "most_corrected_merchants": merchant_counts,
            "most_corrected_categories": category_counts
        }
    except Exception as e:
        logger.error(f"Error getting feedback statistics: {e}")
        return {
            "success": False,
            "error": str(e)
        }

