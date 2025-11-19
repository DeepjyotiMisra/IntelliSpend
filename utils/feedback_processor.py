"""Utility to process feedback and update merchant seed"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


def load_feedback() -> pd.DataFrame:
    """
    Load feedback data from storage.
    
    Returns:
        DataFrame with feedback records
    """
    feedback_path = Path(settings.FEEDBACK_STORAGE_PATH)
    
    if not feedback_path.exists():
        logger.info("No feedback file found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(feedback_path)
        logger.info(f"Loaded {len(df)} feedback records")
        return df
    except Exception as e:
        logger.error(f"Error loading feedback: {e}")
        return pd.DataFrame()


def get_pending_feedback() -> pd.DataFrame:
    """
    Get feedback records that haven't been processed yet.
    
    Returns:
        DataFrame with pending feedback
    """
    df = load_feedback()
    
    if df.empty:
        return df
    
    # Filter for unprocessed feedback
    if 'processed' in df.columns:
        pending = df[df['processed'] == False].copy()
    else:
        # If no processed column, assume all are pending
        pending = df.copy()
        pending['processed'] = False
    
    return pending


def find_duplicate_feedback() -> Dict[str, Any]:
    """
    Find duplicate feedback records based on transaction description, amount, date, and corrections.
    
    Returns:
        Dict with duplicate information and indices
    """
    try:
        feedback_path = Path(settings.FEEDBACK_STORAGE_PATH)
        
        if not feedback_path.exists():
            return {
                "success": False,
                "error": "Feedback file not found",
                "duplicates": [],
                "duplicate_count": 0
            }
        
        # Load feedback
        df = load_feedback()
        
        if df.empty:
            return {
                "success": True,
                "duplicates": [],
                "duplicate_count": 0,
                "message": "No feedback records found"
            }
        
        # Identify duplicates based on key fields
        # Consider duplicates if they have same: transaction_description, amount, date, corrected_merchant, corrected_category
        key_columns = ['transaction_description', 'corrected_merchant', 'corrected_category']
        if 'amount' in df.columns:
            key_columns.append('amount')
        if 'date' in df.columns:
            key_columns.append('date')
        
        # Filter to available columns
        available_key_cols = [col for col in key_columns if col in df.columns]
        
        if not available_key_cols:
            return {
                "success": False,
                "error": "Cannot identify duplicates: missing required columns",
                "duplicates": [],
                "duplicate_count": 0
            }
        
        # Find duplicates
        # Keep first occurrence, mark others as duplicates
        duplicate_mask = df.duplicated(subset=available_key_cols, keep='first')
        duplicate_indices = df[duplicate_mask].index.tolist()
        
        # Group duplicates for display
        duplicate_groups = []
        if duplicate_indices:
            # Group by the key columns
            for idx in duplicate_indices:
                row = df.loc[idx]
                # Find all rows with same key values
                mask = True
                for col in available_key_cols:
                    mask = mask & (df[col] == row[col])
                
                group_indices = df[mask].index.tolist()
                if len(group_indices) > 1:
                    duplicate_groups.append({
                        'indices': group_indices,
                        'transaction': str(row.get('transaction_description', 'N/A')),
                        'corrected_merchant': str(row.get('corrected_merchant', 'N/A')),
                        'corrected_category': str(row.get('corrected_category', 'N/A')),
                        'count': len(group_indices)
                    })
        
        # Remove duplicate groups (keep only unique groups)
        seen = set()
        unique_groups = []
        for group in duplicate_groups:
            group_key = tuple(sorted(group['indices']))
            if group_key not in seen:
                seen.add(group_key)
                unique_groups.append(group)
        
        return {
            "success": True,
            "duplicates": unique_groups,
            "duplicate_count": len(duplicate_indices),
            "duplicate_groups": len(unique_groups),
            "message": f"Found {len(duplicate_indices)} duplicate feedback record(s) in {len(unique_groups)} group(s)"
        }
    
    except Exception as e:
        logger.error(f"Error finding duplicate feedback: {e}")
        return {
            "success": False,
            "error": str(e),
            "duplicates": [],
            "duplicate_count": 0
        }


def remove_duplicate_feedback() -> Dict[str, Any]:
    """
    Remove duplicate feedback records, keeping only the first occurrence.
    
    Returns:
        Dict with success status and removal information
    """
    try:
        feedback_path = Path(settings.FEEDBACK_STORAGE_PATH)
        
        if not feedback_path.exists():
            return {
                "success": False,
                "error": "Feedback file not found"
            }
        
        # Load feedback
        df = load_feedback()
        
        if df.empty:
            return {
                "success": True,
                "message": "No feedback records found",
                "removed_count": 0
            }
        
        # Identify duplicates based on key fields
        key_columns = ['transaction_description', 'corrected_merchant', 'corrected_category']
        if 'amount' in df.columns:
            key_columns.append('amount')
        if 'date' in df.columns:
            key_columns.append('date')
        
        # Filter to available columns
        available_key_cols = [col for col in key_columns if col in df.columns]
        
        if not available_key_cols:
            return {
                "success": False,
                "error": "Cannot identify duplicates: missing required columns"
            }
        
        # Count duplicates before removal
        duplicate_mask = df.duplicated(subset=available_key_cols, keep='first')
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count == 0:
            return {
                "success": True,
                "message": "No duplicates found",
                "removed_count": 0
            }
        
        # Remove duplicates (keep first occurrence)
        df_cleaned = df.drop_duplicates(subset=available_key_cols, keep='first').reset_index(drop=True)
        
        # Save cleaned feedback
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(feedback_path, index=False)
        
        logger.info(f"Removed {duplicate_count} duplicate feedback record(s)")
        
        return {
            "success": True,
            "message": f"Removed {duplicate_count} duplicate feedback record(s)",
            "removed_count": duplicate_count,
            "remaining_count": len(df_cleaned)
        }
    
    except Exception as e:
        logger.error(f"Error removing duplicate feedback: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def delete_feedback(indices: List[int]) -> Dict[str, Any]:
    """
    Delete feedback records by their indices in the CSV.
    
    Args:
        indices: List of row indices to delete (0-based from the full feedback DataFrame)
        
    Returns:
        Dict with success status and message
    """
    try:
        feedback_path = Path(settings.FEEDBACK_STORAGE_PATH)
        
        if not feedback_path.exists():
            return {
                "success": False,
                "error": "Feedback file not found"
            }
        
        # Load feedback
        df = load_feedback()
        
        if df.empty:
            return {
                "success": False,
                "error": "No feedback records found"
            }
        
        # Filter to valid indices
        valid_indices = [idx for idx in indices if 0 <= idx < len(df)]
        
        if not valid_indices:
            return {
                "success": False,
                "error": "No valid indices provided"
            }
        
        # Delete rows
        df = df.drop(df.index[valid_indices]).reset_index(drop=True)
        
        # Save updated feedback
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(feedback_path, index=False)
        
        logger.info(f"Deleted {len(valid_indices)} feedback record(s)")
        
        return {
            "success": True,
            "message": f"Deleted {len(valid_indices)} feedback record(s)",
            "deleted_count": len(valid_indices)
        }
    
    except Exception as e:
        logger.error(f"Error deleting feedback: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def update_merchant_seed_from_feedback(rebuild_index: bool = True) -> Dict[str, Any]:
    """
    Update merchant seed CSV with new patterns from feedback.
    
    Args:
        rebuild_index: Whether to rebuild FAISS index after updating seed
        
    Returns:
        Dict with update results
    """
    try:
        pending_feedback = get_pending_feedback()
        
        if pending_feedback.empty:
            return {
                "success": True,
                "message": "No pending feedback to process",
                "new_patterns": 0,
                "updated_merchants": 0
            }
        
        # Load existing merchant seed
        seed_path = Path(settings.MERCHANT_SEED_PATH)
        if seed_path.exists():
            seed_df = pd.read_csv(seed_path)
            # Check if transaction_pattern column exists, if not use description column
            if 'transaction_pattern' not in seed_df.columns:
                if 'description' in seed_df.columns:
                    seed_df['transaction_pattern'] = seed_df['description']
                elif 'transaction_description' in seed_df.columns:
                    seed_df['transaction_pattern'] = seed_df['transaction_description']
                else:
                    # Create transaction_pattern from merchant name if no description column
                    seed_df['transaction_pattern'] = seed_df['merchant']
        else:
            seed_df = pd.DataFrame(columns=['merchant', 'category', 'transaction_pattern'])
        
        new_patterns = []
        updated_merchants = set()
        
        # Process each feedback record
        for idx, feedback in pending_feedback.iterrows():
            # Handle NaN values and convert to string safely
            corrected_merchant = str(feedback.get('corrected_merchant', '') or '').strip().upper()
            corrected_category = str(feedback.get('corrected_category', '') or '').strip()
            transaction_desc = str(feedback.get('transaction_description', '') or '').strip()
            
            if not corrected_merchant or corrected_merchant == 'UNKNOWN' or corrected_merchant == 'NAN':
                continue
            
            # Normalize transaction description
            from utils.data_utils import normalize_transaction_string
            normalized_pattern = normalize_transaction_string(transaction_desc)
            
            # Check if this pattern already exists (check both transaction_pattern and description columns)
            pattern_col = 'transaction_pattern' if 'transaction_pattern' in seed_df.columns else 'description'
            
            existing = seed_df[
                (seed_df['merchant'] == corrected_merchant) &
                (seed_df[pattern_col] == normalized_pattern)
            ]
            
            if existing.empty:
                # Add new pattern - match the existing seed structure
                new_pattern = {
                    'merchant': corrected_merchant,
                    'category': corrected_category,
                }
                
                # Add the pattern column based on seed structure
                if 'description' in seed_df.columns:
                    new_pattern['description'] = normalized_pattern
                elif 'transaction_pattern' in seed_df.columns:
                    new_pattern['transaction_pattern'] = normalized_pattern
                else:
                    new_pattern['transaction_pattern'] = normalized_pattern
                
                # Add sample_amount if column exists
                if 'sample_amount' in seed_df.columns:
                    amount = feedback.get('amount')
                    # Handle NaN values
                    if pd.isna(amount) or amount is None:
                        new_pattern['sample_amount'] = 0.0
                    else:
                        try:
                            new_pattern['sample_amount'] = float(amount)
                        except (ValueError, TypeError):
                            new_pattern['sample_amount'] = 0.0
                
                new_patterns.append(new_pattern)
                updated_merchants.add(corrected_merchant)
                logger.info(f"Adding new pattern: {corrected_merchant} - {normalized_pattern[:50]}")
        
        # Add new patterns to seed
        if new_patterns:
            new_df = pd.DataFrame(new_patterns)
            seed_df = pd.concat([seed_df, new_df], ignore_index=True)
            
            # Remove duplicates
            seed_df = seed_df.drop_duplicates(subset=['merchant', 'transaction_pattern'], keep='first')
            
            # Save updated seed
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            seed_df.to_csv(seed_path, index=False)
            logger.info(f"Updated merchant seed: Added {len(new_patterns)} new patterns")
        
        # Mark feedback as processed
        feedback_path = Path(settings.FEEDBACK_STORAGE_PATH)
        if feedback_path.exists():
            df = pd.read_csv(feedback_path)
            df.loc[pending_feedback.index, 'processed'] = True
            df.to_csv(feedback_path, index=False)
            logger.info(f"Marked {len(pending_feedback)} feedback records as processed")
        
        # Rebuild FAISS index if requested
        if rebuild_index and new_patterns:
            logger.info("Rebuilding FAISS index with new patterns...")
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "utils/faiss_index_builder.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("FAISS index rebuilt successfully")
            else:
                logger.warning(f"FAISS index rebuild had issues: {result.stderr}")
        
        return {
            "success": True,
            "message": f"Updated merchant seed with {len(new_patterns)} new patterns",
            "new_patterns": len(new_patterns),
            "updated_merchants": len(updated_merchants),
            "merchants": list(updated_merchants)
        }
    
    except Exception as e:
        logger.error(f"Error updating merchant seed from feedback: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_feedback_summary() -> Dict[str, Any]:
    """
    Get summary statistics about feedback.
    
    Returns:
        Dict with feedback summary
    """
    try:
        df = load_feedback()
        
        if df.empty:
            return {
                "total": 0,
                "processed": 0,
                "pending": 0,
                "most_corrected_merchants": {},
                "most_corrected_categories": {}
            }
        
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
            "total": total,
            "processed": int(processed),
            "pending": int(pending),
            "most_corrected_merchants": merchant_counts,
            "most_corrected_categories": category_counts
        }
    except Exception as e:
        logger.error(f"Error getting feedback summary: {e}")
        return {
            "error": str(e)
        }

