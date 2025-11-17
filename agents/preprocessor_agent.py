"""
Preprocessor Agent - Handles data cleaning and validation for IntelliSpend
This agent is responsible for cleaning, normalizing, and validating transaction data
before it's processed by other agents in the pipeline.
"""

from agno.agent import Agent
from agno.tools.file import FileTools
from typing import List, Dict, Any, Optional
import pandas as pd
import re
from datetime import datetime
import logging
from models.transaction import TransactionData
from utils.data_processing import TransactionCleaner

logger = logging.getLogger(__name__)


class PreprocessorAgent(Agent):
    """
    Agent responsible for preprocessing and cleaning transaction data.
    
    Key responsibilities:
    - Data validation and cleaning
    - Amount normalization
    - Date standardization
    - Merchant name cleaning
    - Duplicate detection
    - Data quality scoring
    """
    
    def __init__(self):
        super().__init__(
            name="PreprocessorAgent",
            description="Cleans and validates transaction data with advanced validation and priority tools",
            instructions=[
                "Clean and normalize transaction amounts",
                "Standardize merchant names using custom cleaning tools",
                "Validate date formats and data quality",
                "Remove duplicates and detect anomalies",
                "Score data quality and assign processing priority",
                "Use custom tools for comprehensive data preprocessing"
            ],
            tools=[
                FileTools()  # Built-in file operations
            ]
        )
        self.merchant_normalizations = self._load_merchant_normalizations()
    
    def _load_merchant_normalizations(self) -> Dict[str, str]:
        """Load common merchant name normalizations."""
        return {
            # Common patterns to normalize
            r'(?i)amazon\.com.*': 'Amazon',
            r'(?i)amzn.*': 'Amazon', 
            r'(?i)starbucks.*': 'Starbucks',
            r'(?i)walmart.*': 'Walmart',
            r'(?i)mcdonalds.*': 'McDonald\'s',
            r'(?i)target.*': 'Target',
            r'(?i)costco.*': 'Costco',
            r'(?i)uber.*': 'Uber',
            r'(?i)lyft.*': 'Lyft',
            r'(?i)spotify.*': 'Spotify',
            r'(?i)netflix.*': 'Netflix'
        }
    
    def clean_merchant_name(self, merchant: str) -> str:
        """Clean and normalize merchant names."""
        if not merchant or pd.isna(merchant):
            return "Unknown Merchant"
        
        # Convert to string and strip
        merchant = str(merchant).strip()
        
        # Remove common prefixes/suffixes
        merchant = re.sub(r'^(TST\*|SQ\*|PP\*|PAYPAL\s*\*)', '', merchant, flags=re.IGNORECASE)
        merchant = re.sub(r'\s+\d{2,}/\d{2}$', '', merchant)  # Remove dates
        merchant = re.sub(r'\s+#\d+$', '', merchant)  # Remove reference numbers
        
        # Apply normalizations
        for pattern, replacement in self.merchant_normalizations.items():
            if re.match(pattern, merchant):
                return replacement
        
        # Capitalize properly
        merchant = ' '.join(word.capitalize() for word in merchant.split())
        
        return merchant
    
    def normalize_amount(self, amount: Any) -> float:
        """Normalize transaction amounts."""
        if pd.isna(amount):
            return 0.0
            
        # Convert to string first
        amount_str = str(amount).strip()
        
        # Remove currency symbols and commas
        amount_str = re.sub(r'[^\d.-]', '', amount_str)
        
        try:
            return float(amount_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse amount: {amount}")
            return 0.0
    
    def standardize_date(self, date_val: Any) -> Optional[datetime]:
        """Standardize date formats."""
        if pd.isna(date_val):
            return None
            
        if isinstance(date_val, datetime):
            return date_val
            
        # Try to parse various date formats
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%y',
            '%d-%m-%Y'
        ]
        
        date_str = str(date_val).strip()
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_val}")
        return None
    

    
    def calculate_quality_score(self, transaction: Dict[str, Any]) -> float:
        """Calculate data quality score for a transaction."""
        score = 100.0
        
        # Check required fields
        if not transaction.get('merchant') or transaction.get('merchant') == 'Unknown Merchant':
            score -= 20
        
        if not transaction.get('amount') or transaction.get('amount') == 0:
            score -= 30
            
        if not transaction.get('date'):
            score -= 30
            
        # Check data quality
        if transaction.get('description') and len(transaction.get('description', '')) < 5:
            score -= 10
            
        # Bonus for complete data
        if transaction.get('category'):
            score += 5
            
        return max(0.0, min(100.0, score))
    
    def preprocess_batch(self, transactions: List[TransactionData]) -> List[TransactionData]:
        """
        Batch preprocessing method for parallel processing compatibility.
        
        Args:
            transactions: List of TransactionData objects to preprocess
            
        Returns:
            List of preprocessed TransactionData objects
        """
        try:
            logger.info(f"Batch preprocessing {len(transactions)} transactions")
            
            processed_transactions = []
            
            for i, transaction in enumerate(transactions):
                try:
                    # Clean merchant name
                    cleaned_merchant = self.clean_merchant_name(transaction.merchant_name)
                    
                    # Normalize amount
                    normalized_amount = self.normalize_amount(transaction.amount)
                    
                    # Create new TransactionData with cleaned values
                    processed_transaction = TransactionData(
                        id=transaction.id,
                        transaction_date=transaction.transaction_date,
                        amount=normalized_amount,
                        merchant_name=cleaned_merchant,
                        description=transaction.description
                    )
                    
                    processed_transactions.append(processed_transaction)
                    
                except Exception as e:
                    logger.error(f"Error preprocessing transaction {transaction.id}: {e}")
                    # Keep original transaction if preprocessing fails
                    processed_transactions.append(transaction)
            
            logger.info(f"Successfully batch preprocessed {len(processed_transactions)} transactions")
            return processed_transactions
            
        except Exception as e:
            logger.error(f"Error in batch preprocessing: {e}")
            return transactions  # Return original transactions if batch processing fails

    def process_transactions(self, raw_transactions: List[Dict[str, Any]]) -> List[TransactionData]:
        """Main processing method for cleaning transaction data."""
        try:
            logger.info(f"Processing {len(raw_transactions)} transactions")
            
            processed_transactions = []
            
            for i, transaction in enumerate(raw_transactions):
                try:
                    # Clean and normalize the transaction
                    # Handle both TransactionData objects and dictionaries
                    if hasattr(transaction, 'merchant_name'):
                        # TransactionData object
                        merchant_value = transaction.merchant_name
                        date_value = transaction.transaction_date
                        amount_value = transaction.amount
                        description_value = transaction.description
                        transaction_id = getattr(transaction, 'id', f'txn_{i}')
                        category_value = getattr(transaction, 'category', '')
                    else:
                        # Dictionary format
                        merchant_value = transaction.get('merchant_name') or transaction.get('merchant', '')
                        date_value = transaction.get('transaction_date') or transaction.get('date')
                        amount_value = transaction.get('amount')
                        description_value = transaction.get('description', '')
                        transaction_id = transaction.get('id', f'txn_{i}')
                        category_value = transaction.get('category', '')
                    
                    cleaned_transaction = {
                        'id': transaction_id,
                        'date': self.standardize_date(date_value),
                        'amount': self.normalize_amount(amount_value),
                        'merchant': self.clean_merchant_name(merchant_value),
                        'description': str(description_value).strip(),
                        'category': category_value,
                        'account': getattr(transaction, 'account', '') if hasattr(transaction, 'merchant_name') else transaction.get('account', ''),
                        'transaction_type': getattr(transaction, 'transaction_type', 'debit') if hasattr(transaction, 'merchant_name') else transaction.get('transaction_type', 'debit')
                    }
                    
                    # Calculate quality score
                    quality_score = self.calculate_quality_score(cleaned_transaction)
                    cleaned_transaction['quality_score'] = quality_score
                    
                    # Convert to TransactionData object
                    transaction_data = TransactionData(
                        id=cleaned_transaction['id'],
                        transaction_date=cleaned_transaction['date'] or datetime.now(),
                        amount=cleaned_transaction['amount'],
                        merchant_name=cleaned_transaction['merchant'],
                        description=cleaned_transaction['description']
                    )
                    
                    processed_transactions.append(transaction_data)
                    
                except Exception as e:
                    logger.error(f"Error processing transaction {i}: {e}")
                    continue
            
            # Detect and flag duplicates using centralized cleaner
            if len(processed_transactions) > 1:
                deduplicated = self.cleaner.deduplicate_transactions(processed_transactions)
                # Calculate which transactions were removed
                original_ids = {t.id: i for i, t in enumerate(processed_transactions)}
                remaining_ids = {t.id for t in deduplicated}
                duplicate_indices = [i for tid, i in original_ids.items() if tid not in remaining_ids]
                
                for idx in duplicate_indices:
                    logger.warning(f"Transaction {idx} flagged as potential duplicate")
            else:
                duplicate_indices = []
            
            logger.info(f"Successfully processed {len(processed_transactions)} transactions")
            return processed_transactions
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return []
    
    async def run(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agno agent run method."""
        try:
            processed_transactions = self.process_transactions(transactions)
            
            return {
                "status": "success",
                "processed_count": len(processed_transactions),
                "original_count": len(transactions),
                "transactions": processed_transactions,
                "message": f"Successfully preprocessed {len(processed_transactions)} transactions"
            }
            
        except Exception as e:
            logger.error(f"Preprocessor agent error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transactions": []
            }