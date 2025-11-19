"""Helper functions for feedback collection and processing"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from config.settings import settings
from agents.tools import store_feedback
import logging

logger = logging.getLogger(__name__)


def submit_feedback_from_result(
    result: Dict[str, Any],
    corrected_merchant: str,
    corrected_category: str,
    use_llm: bool = True,
    update_csv: bool = True
) -> Dict[str, Any]:
    """
    Submit feedback for a transaction result.
    
    Args:
        result: Transaction result dict from pipeline
        corrected_merchant: User-corrected merchant name
        corrected_category: User-corrected category
        use_llm: Whether to use LLM agent for processing (default: True)
        update_csv: Whether to update categorized_transactions.csv (default: True)
        
    Returns:
        Feedback storage result
    """
    try:
        # Store feedback first
        feedback_result = store_feedback(
            transaction_description=result.get('original_description', ''),
            original_merchant=result.get('merchant', 'UNKNOWN'),
            original_category=result.get('category', 'Other'),
            corrected_merchant=corrected_merchant.strip().upper(),
            corrected_category=corrected_category.strip(),
            amount=result.get('amount'),
            date=result.get('date'),
            confidence_score=result.get('confidence_score', 0.0)
        )
        
        # Update categorized_transactions.csv if requested
        if update_csv and feedback_result.get('success'):
            update_result = update_categorized_transaction(
                transaction_description=result.get('original_description', ''),
                corrected_merchant=corrected_merchant.strip().upper(),
                corrected_category=corrected_category.strip(),
                amount=result.get('amount'),
                date=result.get('date')
            )
            feedback_result['csv_updated'] = update_result.get('updated', False)
            if update_result.get('updated'):
                feedback_result['updated_count'] = update_result.get('updated_count', 0)
            else:
                feedback_result['csv_update_error'] = update_result.get('error', 'Unknown error')
        
        # Optionally process with LLM agent (but don't store feedback again - it's already stored)
        # The LLM agent's process_feedback function calls store_feedback, but we've already stored it
        # So we skip LLM processing if feedback was already stored, or we need to prevent duplicate storage
        if use_llm and feedback_result.get('success') and not feedback_result.get('duplicate', False):
            try:
                from agents.feedback_agent import process_feedback
                # This will use the current FEEDBACK_MODEL_PROVIDER from environment
                # Note: process_feedback will try to store feedback again, but store_feedback now prevents duplicates
                llm_response = process_feedback(
                    transaction_description=result.get('original_description', ''),
                    original_merchant=result.get('merchant', 'UNKNOWN'),
                    original_category=result.get('category', 'Other'),
                    corrected_merchant=corrected_merchant.strip().upper(),
                    corrected_category=corrected_category.strip(),
                    amount=result.get('amount'),
                    date=result.get('date'),
                    confidence_score=result.get('confidence_score', 0.0)
                )
                feedback_result['llm_processed'] = True
                feedback_result['llm_response'] = llm_response[:200]  # First 200 chars
            except Exception as e:
                logger.warning(f"LLM processing failed, but feedback stored: {e}")
                feedback_result['llm_processed'] = False
        
        return feedback_result
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def update_categorized_transaction(
    transaction_description: str,
    corrected_merchant: str,
    corrected_category: str,
    amount: float = None,
    date: str = None,
    file_path: str = None
) -> Dict[str, Any]:
    """
    Update a transaction in the categorized_transactions.csv file.
    Uses exact matching with description, and optionally amount and date for unique identification.
    
    Args:
        transaction_description: Original transaction description to match
        corrected_merchant: Corrected merchant name
        corrected_category: Corrected category
        amount: Transaction amount (optional, for more precise matching)
        date: Transaction date (optional, for more precise matching)
        file_path: Path to categorized transactions CSV
        
    Returns:
        Dict with update result
    """
    if file_path is None:
        file_path = 'output/categorized_transactions.csv'
    
    path = Path(file_path)
    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "updated": False
        }
    
    try:
        df = pd.read_csv(path)
        
        # Normalize description for matching (case-insensitive, whitespace-insensitive)
        normalized_desc = transaction_description.strip().upper()
        
        # Start with description matching
        desc_col = None
        if 'original_description' in df.columns:
            desc_col = 'original_description'
        elif 'description' in df.columns:
            desc_col = 'description'
        else:
            return {
                "success": False,
                "error": "No description column found in CSV",
                "updated": False
            }
        
        # Exact match on description
        mask = df[desc_col].str.strip().str.upper() == normalized_desc
        
        # If amount is provided, also match on amount for more precision
        if amount is not None and 'amount' in df.columns:
            # Handle both positive and negative amounts (some CSVs have negative amounts)
            amount_mask = (df['amount'].abs() == abs(float(amount))) | (df['amount'] == float(amount))
            mask = mask & amount_mask
        
        # If date is provided, also match on date for more precision
        if date is not None and 'date' in df.columns:
            # Normalize date format for comparison
            try:
                date_str = str(date).strip()
                date_mask = df['date'].astype(str).str.strip() == date_str
                mask = mask & date_mask
            except Exception as e:
                logger.debug(f"Could not match on date: {e}")
        
        # Only update if we have exactly one match
        match_count = mask.sum()
        
        if match_count == 0:
            return {
                "success": False,
                "error": "Transaction not found in categorized_transactions.csv",
                "updated": False
            }
        elif match_count > 1:
            # Multiple matches - this shouldn't happen with exact matching
            # But if it does, we'll update all matches (though this is a warning)
            logger.warning(f"Found {match_count} matching transactions, updating all of them")
        
        # Update the matching row(s) - but only the first one if multiple matches
        if match_count == 1:
            # Single match - update it
            df.loc[mask, 'merchant'] = corrected_merchant
            df.loc[mask, 'category'] = corrected_category
            # Update confidence score to indicate manual correction
            if 'confidence_score' in df.columns:
                df.loc[mask, 'confidence_score'] = 1.0  # High confidence for manual corrections
            # Add a note that this was manually corrected
            if 'classification_source' in df.columns:
                df.loc[mask, 'classification_source'] = 'manual_correction'
        else:
            # Multiple matches - update only the first one
            first_match_idx = df[mask].index[0]
            df.loc[first_match_idx, 'merchant'] = corrected_merchant
            df.loc[first_match_idx, 'category'] = corrected_category
            if 'confidence_score' in df.columns:
                df.loc[first_match_idx, 'confidence_score'] = 1.0
            if 'classification_source' in df.columns:
                df.loc[first_match_idx, 'classification_source'] = 'manual_correction'
            logger.warning(f"Multiple matches found, updated only the first one (index {first_match_idx})")
        
        # Save updated DataFrame
        df.to_csv(path, index=False)
        
        updated_count = 1 if match_count > 1 else match_count
        logger.info(f"Updated {updated_count} transaction(s) in {file_path}")
        
        return {
            "success": True,
            "updated": True,
            "updated_count": updated_count,
            "message": f"Updated {updated_count} transaction(s)"
        }
            
    except Exception as e:
        logger.error(f"Error updating categorized transaction: {e}")
        return {
            "success": False,
            "error": str(e),
            "updated": False
        }


def load_categorized_transactions(file_path: str = None) -> pd.DataFrame:
    """
    Load categorized transactions from output file.
    
    Args:
        file_path: Path to categorized transactions CSV
        
    Returns:
        DataFrame with categorized transactions
    """
    if file_path is None:
        file_path = 'output/categorized_transactions.csv'
    
    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"Error loading categorized transactions: {e}")
        return pd.DataFrame()

