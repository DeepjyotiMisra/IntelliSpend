"""
Data Processing Utilities for IntelliSpend

Utilities for parsing, cleaning, and preprocessing financial transaction data
from various sources (CSV, JSON, bank APIs, etc.)
"""

import os
import csv
import json
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path

from models.transaction import TransactionData, TransactionType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionParser:
    """
    Utility class for parsing transaction data from various formats
    """
    
    def __init__(self):
        """Initialize the transaction parser"""
        self.supported_formats = ['csv', 'json', 'excel', 'tsv']
        
        # Common field mappings for different bank formats
        self.field_mappings = {
            'amount': ['amount', 'Amount', 'AMOUNT', 'transaction_amount', 'value', 'debit', 'credit'],
            'merchant_name': ['merchant', 'Merchant', 'MERCHANT', 'description', 'Description', 'DESCRIPTION', 
                            'payee', 'Payee', 'PAYEE', 'vendor', 'merchant_name'],
            'transaction_date': ['date', 'Date', 'DATE', 'transaction_date', 'posting_date', 'trans_date'],
            'description': ['description', 'Description', 'DESCRIPTION', 'memo', 'Memo', 'MEMO', 'details'],
            'account_id': ['account', 'Account', 'ACCOUNT', 'account_id', 'account_number'],
            'reference_number': ['reference', 'Reference', 'ref_number', 'transaction_id', 'id'],
            'currency': ['currency', 'Currency', 'CURRENCY', 'curr']
        }
    
    def parse_csv_file(self, 
                      file_path: str, 
                      delimiter: str = ',',
                      encoding: str = 'utf-8',
                      skip_rows: int = 0) -> List[TransactionData]:
        """
        Parse transactions from CSV file
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter character
            encoding: File encoding
            skip_rows: Number of rows to skip at the beginning
            
        Returns:
            List of TransactionData objects
        """
        logger.info(f"Parsing CSV file: {file_path}")
        
        transactions = []
        
        try:
            # Read CSV with pandas for better handling
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, skiprows=skip_rows)
            
            # Auto-detect field mappings
            field_map = self._auto_detect_fields(df.columns.tolist())
            logger.info(f"Detected field mappings: {field_map}")
            
            # Convert each row to TransactionData
            for idx, row in df.iterrows():
                try:
                    transaction = self._row_to_transaction(row, field_map)
                    if transaction:
                        transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Error parsing row {idx}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(transactions)} transactions from CSV")
            return transactions
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return []
    
    def parse_json_file(self, file_path: str) -> List[TransactionData]:
        """
        Parse transactions from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of TransactionData objects
        """
        logger.info(f"Parsing JSON file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            transactions = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of transaction objects
                transaction_list = data
            elif isinstance(data, dict):
                # Check common wrapper keys
                for key in ['transactions', 'data', 'records', 'results']:
                    if key in data and isinstance(data[key], list):
                        transaction_list = data[key]
                        break
                else:
                    # Single transaction object
                    transaction_list = [data]
            else:
                raise ValueError("Unsupported JSON structure")
            
            # Convert each object to TransactionData
            for idx, item in enumerate(transaction_list):
                try:
                    transaction = self._dict_to_transaction(item)
                    if transaction:
                        transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Error parsing JSON item {idx}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(transactions)} transactions from JSON")
            return transactions
            
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return []
    
    def parse_excel_file(self, 
                        file_path: str,
                        sheet_name: Union[str, int] = 0,
                        skip_rows: int = 0) -> List[TransactionData]:
        """
        Parse transactions from Excel file
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index
            skip_rows: Number of rows to skip
            
        Returns:
            List of TransactionData objects
        """
        logger.info(f"Parsing Excel file: {file_path}")
        
        try:
            # Read Excel with pandas
            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows)
            
            # Auto-detect field mappings
            field_map = self._auto_detect_fields(df.columns.tolist())
            logger.info(f"Detected field mappings: {field_map}")
            
            transactions = []
            
            # Convert each row to TransactionData
            for idx, row in df.iterrows():
                try:
                    transaction = self._row_to_transaction(row, field_map)
                    if transaction:
                        transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Error parsing row {idx}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(transactions)} transactions from Excel")
            return transactions
            
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            return []
    
    def _auto_detect_fields(self, columns: List[str]) -> Dict[str, str]:
        """
        Auto-detect field mappings based on column names
        
        Args:
            columns: List of column names
            
        Returns:
            Mapping of standard field names to actual column names
        """
        field_map = {}
        columns_lower = [col.lower() for col in columns]
        
        for standard_field, possible_names in self.field_mappings.items():
            for possible_name in possible_names:
                if possible_name.lower() in columns_lower:
                    # Find the actual column name (preserve case)
                    actual_column = columns[columns_lower.index(possible_name.lower())]
                    field_map[standard_field] = actual_column
                    break
        
        return field_map
    
    def _row_to_transaction(self, row: pd.Series, field_map: Dict[str, str]) -> Optional[TransactionData]:
        """
        Convert pandas row to TransactionData object
        
        Args:
            row: Pandas Series representing a row
            field_map: Field mapping dictionary
            
        Returns:
            TransactionData object or None if conversion fails
        """
        try:
            # Extract amount
            amount = 0.0
            if 'amount' in field_map:
                amount_str = str(row.get(field_map['amount'], 0))
                amount = self._parse_amount(amount_str)
            
            # Extract merchant name
            merchant_name = ""
            if 'merchant_name' in field_map:
                merchant_name = str(row.get(field_map['merchant_name'], "")).strip()
            
            # Extract date
            transaction_date = datetime.now()
            if 'transaction_date' in field_map:
                date_str = str(row.get(field_map['transaction_date'], ""))
                transaction_date = self._parse_date(date_str)
            
            # Extract description
            description = ""
            if 'description' in field_map:
                description = str(row.get(field_map['description'], "")).strip()
            
            # If merchant_name is empty, try to use description
            if not merchant_name and description:
                merchant_name = description
            
            # Skip if essential fields are missing
            if not merchant_name or amount == 0:
                return None
            
            # Create transaction
            transaction = TransactionData(
                amount=abs(amount),  # Use absolute value
                merchant_name=merchant_name,
                description=description,
                transaction_date=transaction_date,
                transaction_type=TransactionType.DEBIT if amount < 0 else TransactionType.CREDIT,
                currency=str(row.get(field_map.get('currency', ''), 'USD')).strip() or 'USD',
                account_id=str(row.get(field_map.get('account_id', ''), '')).strip() or None,
                reference_number=str(row.get(field_map.get('reference_number', ''), '')).strip() or None
            )
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error converting row to transaction: {e}")
            return None
    
    def _dict_to_transaction(self, data: Dict[str, Any]) -> Optional[TransactionData]:
        """
        Convert dictionary to TransactionData object
        
        Args:
            data: Dictionary containing transaction data
            
        Returns:
            TransactionData object or None if conversion fails
        """
        try:
            # Auto-detect field mappings from dictionary keys
            field_map = self._auto_detect_fields(list(data.keys()))
            
            # Create pandas Series for consistent processing
            row = pd.Series(data)
            
            return self._row_to_transaction(row, field_map)
            
        except Exception as e:
            logger.error(f"Error converting dict to transaction: {e}")
            return None
    
    def _parse_amount(self, amount_str: str) -> float:
        """
        Parse amount string to float, handling various formats
        
        Args:
            amount_str: String representation of amount
            
        Returns:
            Float amount
        """
        if not amount_str or pd.isna(amount_str):
            return 0.0
        
        # Clean the string
        amount_str = str(amount_str).strip()
        
        # Remove common currency symbols and characters
        amount_str = re.sub(r'[$£€¥₹,\s]', '', amount_str)
        
        # Handle parentheses (negative amounts)
        if amount_str.startswith('(') and amount_str.endswith(')'):
            amount_str = '-' + amount_str[1:-1]
        
        try:
            return float(amount_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse amount: {amount_str}")
            return 0.0
    
    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse date string to datetime object, trying multiple formats
        
        Args:
            date_str: String representation of date
            
        Returns:
            Datetime object
        """
        if not date_str or pd.isna(date_str):
            return datetime.now()
        
        date_str = str(date_str).strip()
        
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y%m%d',
            '%m-%d-%Y',
            '%d-%m-%Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try pandas date parser as fallback
        try:
            return pd.to_datetime(date_str)
        except:
            logger.warning(f"Could not parse date: {date_str}")
            return datetime.now()


class TransactionCleaner:
    """
    Utility class for cleaning and normalizing transaction data
    """
    
    def __init__(self):
        """Initialize the transaction cleaner"""
        self.merchant_aliases = self._load_merchant_aliases()
        self.stop_words = {'inc', 'corp', 'company', 'co', 'ltd', 'llc', 'the', 'and', '&'}
    
    def clean_merchant_name(self, merchant_name: str) -> str:
        """
        Clean and normalize merchant name
        
        Args:
            merchant_name: Original merchant name
            
        Returns:
            Cleaned merchant name
        """
        if not merchant_name:
            return ""
        
        # Convert to string and strip
        name = str(merchant_name).strip()
        
        # Remove common prefixes/suffixes from card processors
        name = re.sub(r'^(SQ\s*\*|TST\s*\*|PAYPAL\s*\*|VENMO\s*\*)', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*\*.*$', '', name)  # Remove trailing * and everything after
        
        # Remove location information in parentheses or after #
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'\s*#.*$', '', name)
        
        # Remove common suffixes
        name = re.sub(r'\s*(INC|CORP|COMPANY|CO|LTD|LLC)\.?\s*$', '', name, flags=re.IGNORECASE)
        
        # Clean whitespace and special characters
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
        name = re.sub(r'[^\w\s\-&\.]', '', name)  # Keep only alphanumeric, spaces, hyphens, ampersands, dots
        
        # Check for known aliases
        name_normalized = name.upper().strip()
        if name_normalized in self.merchant_aliases:
            return self.merchant_aliases[name_normalized]
        
        # Title case for presentation
        return name.strip().title()
    
    def clean_description(self, description: str) -> str:
        """
        Clean and normalize transaction description
        
        Args:
            description: Original description
            
        Returns:
            Cleaned description
        """
        if not description:
            return ""
        
        desc = str(description).strip()
        
        # Remove common card processor prefixes
        desc = re.sub(r'^(DEBIT|CREDIT|PURCHASE|WITHDRAWAL|DEPOSIT)\s*-?\s*', '', desc, flags=re.IGNORECASE)
        
        # Remove reference numbers and IDs
        desc = re.sub(r'\b\d{6,}\b', '', desc)  # Remove long numbers (likely IDs)
        desc = re.sub(r'\bREF\s*#?\s*\d+', '', desc, flags=re.IGNORECASE)
        
        # Clean whitespace
        desc = re.sub(r'\s+', ' ', desc).strip()
        
        return desc
    
    def deduplicate_transactions(self, 
                               transactions: List[TransactionData],
                               time_window_minutes: int = 5,
                               amount_tolerance: float = 0.01) -> List[TransactionData]:
        """
        Remove duplicate transactions based on merchant, amount, and time proximity
        
        Args:
            transactions: List of transactions to deduplicate
            time_window_minutes: Time window for considering duplicates (minutes)
            amount_tolerance: Amount tolerance for duplicates
            
        Returns:
            Deduplicated list of transactions
        """
        logger.info(f"Deduplicating {len(transactions)} transactions")
        
        if not transactions:
            return []
        
        # Sort by date
        sorted_transactions = sorted(transactions, key=lambda t: t.transaction_date)
        deduplicated = []
        
        for current in sorted_transactions:
            is_duplicate = False
            
            # Check against recent transactions
            for recent in reversed(deduplicated[-10:]):  # Check last 10 transactions
                # Time difference check
                time_diff = abs((current.transaction_date - recent.transaction_date).total_seconds() / 60)
                if time_diff > time_window_minutes:
                    continue
                
                # Amount check
                amount_diff = abs(current.amount - recent.amount)
                if amount_diff > amount_tolerance:
                    continue
                
                # Merchant similarity check
                if self._merchants_similar(current.merchant_name, recent.merchant_name):
                    is_duplicate = True
                    logger.debug(f"Duplicate found: {current.merchant_name} ${current.amount}")
                    break
            
            if not is_duplicate:
                deduplicated.append(current)
        
        removed_count = len(transactions) - len(deduplicated)
        logger.info(f"Removed {removed_count} duplicate transactions")
        
        return deduplicated
    
    def _merchants_similar(self, merchant1: str, merchant2: str) -> bool:
        """
        Check if two merchant names are similar enough to be considered the same
        
        Args:
            merchant1: First merchant name
            merchant2: Second merchant name
            
        Returns:
            True if merchants are similar
        """
        if not merchant1 or not merchant2:
            return False
        
        # Normalize for comparison
        m1 = re.sub(r'\W+', '', merchant1.lower())
        m2 = re.sub(r'\W+', '', merchant2.lower())
        
        # Exact match
        if m1 == m2:
            return True
        
        # Check if one is contained in the other (for partial matches)
        if len(m1) >= 5 and len(m2) >= 5:
            if m1 in m2 or m2 in m1:
                return True
        
        return False
    
    def _load_merchant_aliases(self) -> Dict[str, str]:
        """
        Load merchant name aliases for normalization
        
        Returns:
            Dictionary mapping aliases to canonical names
        """
        return {
            'AMAZON.COM': 'Amazon',
            'AMAZON MARKETPLACE': 'Amazon',
            'AMZ': 'Amazon',
            'STARBUCKS COFFEE': 'Starbucks',
            'STARBUCKS': 'Starbucks',
            'SBUX': 'Starbucks',
            'MCDONALD\'S': 'McDonalds',
            'MCDONALDS': 'McDonalds',
            'MCD': 'McDonalds',
            'WAL-MART': 'Walmart',
            'WALMART': 'Walmart',
            'WMT': 'Walmart',
            'TARGET STORES': 'Target',
            'TARGET': 'Target',
            'TGT': 'Target',
            'SHELL OIL': 'Shell',
            'SHELL': 'Shell',
            'CHEVRON': 'Chevron',
            'EXXON MOBIL': 'Exxon Mobil',
            'BP': 'BP'
        }


def load_sample_transactions(file_path: str = "data/sample_transactions.csv") -> List[TransactionData]:
    """
    Load sample transactions for testing and development
    
    Args:
        file_path: Path to sample data file
        
    Returns:
        List of TransactionData objects
    """
    parser = TransactionParser()
    
    if not os.path.exists(file_path):
        logger.warning(f"Sample file not found: {file_path}")
        return _generate_sample_transactions()
    
    # Determine file type and parse accordingly
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        return parser.parse_csv_file(file_path)
    elif file_ext == '.json':
        return parser.parse_json_file(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        return parser.parse_excel_file(file_path)
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        return []


def _generate_sample_transactions() -> List[TransactionData]:
    """
    Generate sample transactions for testing
    
    Returns:
        List of sample TransactionData objects
    """
    logger.info("Generating sample transactions")
    
    sample_data = [
        {"amount": 4.75, "merchant": "Starbucks Coffee", "description": "Coffee purchase"},
        {"amount": 45.99, "merchant": "Shell Gas Station", "description": "Fuel"},
        {"amount": 67.23, "merchant": "Safeway", "description": "Groceries"},
        {"amount": 12.99, "merchant": "Netflix", "description": "Monthly subscription"},
        {"amount": 89.99, "merchant": "Amazon.com", "description": "Online shopping"},
        {"amount": 25.50, "merchant": "McDonald's", "description": "Food order"},
        {"amount": 156.78, "merchant": "Pacific Gas & Electric", "description": "Utility bill"},
        {"amount": 75.00, "merchant": "Uber", "description": "Ride fare"},
        {"amount": 29.99, "merchant": "Best Buy", "description": "Electronics"},
        {"amount": 8.50, "merchant": "Subway", "description": "Lunch"}
    ]
    
    transactions = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i, data in enumerate(sample_data):
        transaction = TransactionData(
            amount=data["amount"],
            merchant_name=data["merchant"],
            description=data["description"],
            transaction_date=base_date + timedelta(days=i*3)
        )
        transactions.append(transaction)
    
    return transactions


if __name__ == "__main__":
    # Test data processing utilities
    
    # Generate sample transactions
    transactions = _generate_sample_transactions()
    print(f"Generated {len(transactions)} sample transactions")
    
    # Test cleaning
    cleaner = TransactionCleaner()
    
    for transaction in transactions[:3]:
        print(f"Original: {transaction.merchant_name}")
        cleaned = cleaner.clean_merchant_name(transaction.merchant_name)
        print(f"Cleaned: {cleaned}")
        print("-" * 40)