"""Data utilities for loading and processing transaction data"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_transaction_string(text: str) -> str:
    """
    Normalize transaction description string.
    
    Args:
        text: Raw transaction description
        
    Returns:
        Normalized transaction string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to uppercase for consistency
    normalized = text.upper().strip()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove special characters that don't add meaning (keep alphanumeric, spaces, *, -, /)
    normalized = re.sub(r'[^\w\s*\-/]', ' ', normalized)
    
    # Remove transaction numbers (common patterns like TXN 12345)
    normalized = re.sub(r'\bTXN\s+\d+\b', '', normalized)
    normalized = re.sub(r'\b\d{5,}\b', '', normalized)  # Remove long number sequences
    
    # Clean up multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def load_merchant_seed_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load merchant seed data from CSV file.
    
    Args:
        file_path: Path to merchants_seed.csv. If None, uses default from settings.
        
    Returns:
        DataFrame with columns: merchant, description, sample_amount, category
    """
    from config.settings import settings
    
    if file_path is None:
        file_path = settings.MERCHANT_SEED_PATH
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Merchant seed file not found: {file_path}\n"
            f"Please create {file_path} with columns: merchant, description, sample_amount, category"
        )
    
    try:
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = ['merchant', 'description', 'category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Normalize descriptions
        df['description_normalized'] = df['description'].apply(normalize_transaction_string)
        
        logger.info(f"Loaded {len(df)} merchant seed records from {file_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading merchant seed data: {e}")
        raise


def load_transactions(file_path: str, 
                     description_col: str = 'description',
                     amount_col: Optional[str] = 'amount',
                     date_col: Optional[str] = 'date') -> pd.DataFrame:
    """
    Load transaction data from CSV file.
    
    Args:
        file_path: Path to transactions CSV file
        description_col: Name of the column containing transaction descriptions
        amount_col: Name of the amount column (optional)
        date_col: Name of the date column (optional)
        
    Returns:
        DataFrame with normalized transaction data
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {file_path}")
    
    try:
        df = pd.read_csv(path)
        
        # Validate description column exists
        if description_col not in df.columns:
            raise ValueError(f"Description column '{description_col}' not found in {file_path}")
        
        # Normalize descriptions
        df['description_normalized'] = df[description_col].apply(normalize_transaction_string)
        
        # Ensure description_normalized is not empty
        df = df[df['description_normalized'].str.len() > 0].copy()
        
        logger.info(f"Loaded {len(df)} transactions from {file_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading transactions: {e}")
        raise


def validate_transaction_data(df: pd.DataFrame, 
                              description_col: str = 'description') -> Tuple[bool, List[str]]:
    """
    Validate transaction DataFrame structure and data quality.
    
    Args:
        df: DataFrame to validate
        description_col: Name of description column
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    if description_col not in df.columns:
        errors.append(f"Required column '{description_col}' not found")
    
    # Check for null descriptions
    if description_col in df.columns:
        null_count = df[description_col].isna().sum()
        if null_count > 0:
            errors.append(f"Found {null_count} null values in '{description_col}' column")
        
        empty_count = (df[description_col].astype(str).str.strip() == '').sum()
        if empty_count > 0:
            errors.append(f"Found {empty_count} empty values in '{description_col}' column")
    
    return len(errors) == 0, errors


def prepare_transaction_batch(descriptions: List[str]) -> List[str]:
    """
    Prepare a batch of transaction descriptions for processing.
    
    Args:
        descriptions: List of raw transaction descriptions
        
    Returns:
        List of normalized transaction descriptions
    """
    return [normalize_transaction_string(desc) for desc in descriptions]


def extract_payment_mode(description: str) -> str:
    """
    Extract payment mode from transaction description.
    
    Args:
        description: Transaction description
        
    Returns:
        Payment mode (UPI, NEFT, IMPS, CARD, UNKNOWN)
    """
    desc_lower = description.lower()
    
    if 'upi' in desc_lower:
        return 'UPI'
    elif 'neft' in desc_lower:
        return 'NEFT'
    elif 'imps' in desc_lower:
        return 'IMPS'
    elif 'pos' in desc_lower or 'card' in desc_lower:
        return 'CARD'
    elif 'rtgs' in desc_lower:
        return 'RTGS'
    else:
        return 'UNKNOWN'


def create_sample_transactions() -> pd.DataFrame:
    """
    Create a sample transactions DataFrame for testing.
    
    Returns:
        DataFrame with sample transaction data
    """
    sample_data = {
        'description': [
            'AMAZON PAY INDIA TXN 12345',
            'UBER TRIP MUMBAI 67890',
            'STARBUCKS COFFEE MUMBAI',
            'ZOMATO ORDER 98765',
            'SHELL PETROL PUMP MUMBAI',
            'NETFLIX SUBSCRIPTION',
            'AIRTEL BILL PAYMENT',
            'FLIPKART PURCHASE TXN 11111'
        ],
        'amount': [1500.00, 250.00, 450.00, 650.00, 2000.00, 799.00, 599.00, 3000.00],
        'date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', 
                 '2025-01-19', '2025-01-20', '2025-01-21', '2025-01-22']
    }
    
    df = pd.DataFrame(sample_data)
    df['description_normalized'] = df['description'].apply(normalize_transaction_string)
    
    return df

