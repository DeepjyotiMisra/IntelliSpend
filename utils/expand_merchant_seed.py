"""Expand merchants_seed.csv with multiple transaction patterns per merchant"""

import sys
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import normalize_transaction_string
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_merchant_from_description(description: str) -> str:
    """Extract merchant name from transaction description"""
    desc_upper = description.upper()
    
    # Comprehensive merchant patterns
    merchant_patterns = {
        'AMAZON': ['AMAZON', 'AMZ'],
        'FLIPKART': ['FLIPKART', 'ECART'],
        'UBER': ['UBER'],
        'OLA': ['OLA'],
        'ZOMATO': ['ZOMATO'],
        'SWIGGY': ['SWIGGY'],
        'BOOKMYSHOW': ['BOOKMYSHOW'],
        'NYKAA': ['NYKAA'],
        'MYNTRA': ['MYNTRA'],
        'PHONEPE': ['PHONEPE'],
        'PAYTM': ['PAYTM'],
        'GOOGLE': ['GOOGLE'],
        'APPLE': ['APPLE'],
        'NETFLIX': ['NETFLIX'],
        'SPOTIFY': ['SPOTIFY'],
        'STARBUCKS': ['STARBUCKS'],
        'KFC': ['KFC'],
        'MCDONALDS': ['MCDONALDS', 'MCD'],
        'DOMINOS': ['DOMINOS'],
        'SHELL': ['SHELL PETROL', 'SHELL'],
        'HP': ['HPCL', 'HP PETROL', 'HP'],
        'INDIAN OIL': ['INDIAN OIL', 'INDIANOIL'],
        'AIRINDIA': ['AIRINDIA', 'AIR INDIA'],
        'IRCTC': ['IRCTC'],
        'MAKEMYTRIP': ['MAKEMYTRIP', 'MAKE MY TRIP'],
        'JIOMART': ['JIOMART', 'JIO MART'],
        'BIGBASKET': ['BIGBASKET', 'BIG BASKET'],
        'HDFC': ['HDFC BANK', 'HDFCBANK', 'HDFC'],
        'ICICI': ['ICICI BANK', 'ICICIBANK', 'ICICI'],
        'SBI': ['SBI ATM', 'SBI'],
    }
    
    # Check patterns
    for merchant, patterns in merchant_patterns.items():
        for pattern in patterns:
            if pattern in desc_upper:
                return merchant
    
    # Try "TRANSFER TO X" pattern
    transfer_match = re.search(r'TRANSFER TO ([A-Z\s]+)', desc_upper)
    if transfer_match:
        merchant = transfer_match.group(1).strip()
        merchant = re.sub(r'\s+\d+$', '', merchant)
        # Map to known merchants
        for known_merchant, patterns in merchant_patterns.items():
            for pattern in patterns:
                if pattern in merchant:
                    return known_merchant
        return merchant
    
    # Try "X TXN" or "X DELHI/MUMBAI" patterns
    txn_match = re.search(r'^([A-Z\s]+?)\s+(?:DELHI|MUMBAI|IN|TXN)', desc_upper)
    if txn_match:
        merchant = txn_match.group(1).strip()
        for known_merchant, patterns in merchant_patterns.items():
            for pattern in patterns:
                if pattern in merchant:
                    return known_merchant
        return merchant
    
    # Try UPI pattern: UPI/xxx/MERCHANT
    upi_match = re.search(r'UPI/\d+/([A-Z]+)', desc_upper)
    if upi_match:
        merchant_name = upi_match.group(1)
        for known_merchant, patterns in merchant_patterns.items():
            for pattern in patterns:
                if pattern in merchant_name or merchant_name in pattern:
                    return known_merchant
        return merchant_name
    
    return None


def categorize_merchant(merchant: str, description: str) -> str:
    """Categorize merchant based on name and description"""
    merchant_upper = merchant.upper() if merchant else ''
    desc_upper = description.upper()
    
    # Shopping
    if merchant_upper in ['AMAZON', 'FLIPKART', 'MYNTRA', 'NYKAA', 'JIOMART', 'BIGBASKET']:
        return 'Shopping'
    
    # Transport
    if merchant_upper in ['UBER', 'OLA']:
        return 'Transport'
    if 'PETROL' in desc_upper or 'FUEL' in desc_upper or merchant_upper in ['SHELL', 'HP', 'INDIAN OIL', 'HPCL']:
        return 'Transport'
    
    # Food & Dining
    if merchant_upper in ['ZOMATO', 'SWIGGY', 'STARBUCKS', 'KFC', 'MCDONALDS', 'DOMINOS']:
        return 'Food & Dining'
    
    # Entertainment
    if merchant_upper in ['NETFLIX', 'SPOTIFY', 'BOOKMYSHOW']:
        return 'Entertainment'
    
    # Travel
    if merchant_upper in ['AIRINDIA', 'IRCTC', 'MAKEMYTRIP']:
        return 'Travel'
    
    # Utilities
    if 'BILL' in desc_upper or 'RECHARGE' in desc_upper:
        return 'Utilities'
    
    # Banking
    if merchant_upper in ['HDFC', 'ICICI', 'SBI'] or 'ATM' in desc_upper:
        return 'Banking'
    
    # Bills & Payments
    if merchant_upper in ['PHONEPE', 'PAYTM', 'GOOGLE', 'APPLE']:
        return 'Bills & Payments'
    
    return 'Other'


def expand_merchant_seed(transactions_file: str = 'data/raw_transactions.csv',
                        output_file: str = 'data/merchants_seed.csv',
                        min_occurrences: int = 2,
                        max_patterns_per_merchant: int = 10) -> pd.DataFrame:
    """
    Expand merchant seed with multiple transaction patterns per merchant.
    
    Args:
        transactions_file: Path to raw transactions CSV
        output_file: Path to save expanded merchants seed CSV
        min_occurrences: Minimum occurrences to include merchant
        max_patterns_per_merchant: Maximum patterns to include per merchant
        
    Returns:
        DataFrame with expanded merchant seed data
    """
    logger.info(f"Loading transactions from {transactions_file}...")
    df = pd.read_csv(transactions_file)
    
    # Group transactions by merchant
    merchant_data = defaultdict(lambda: {
        'descriptions': [],
        'amounts': [],
        'count': 0,
        'patterns': defaultdict(int)  # Track unique patterns
    })
    
    logger.info("Extracting merchants from transactions...")
    for _, row in df.iterrows():
        desc = str(row.get('description', ''))
        amount = row.get('amount', 0.0)
        
        merchant = extract_merchant_from_description(desc)
        if merchant and merchant != 'UNKNOWN':
            normalized = normalize_transaction_string(desc)
            merchant_data[merchant]['descriptions'].append(desc)
            merchant_data[merchant]['amounts'].append(amount)
            merchant_data[merchant]['count'] += 1
            merchant_data[merchant]['patterns'][normalized] += 1
    
    # Filter by minimum occurrences
    filtered_merchants = {k: v for k, v in merchant_data.items() 
                        if v['count'] >= min_occurrences}
    
    logger.info(f"Found {len(filtered_merchants)} unique merchants (min {min_occurrences} occurrences)")
    
    # Create expanded seed data with multiple patterns per merchant
    seed_records = []
    
    for merchant, data in sorted(filtered_merchants.items(), key=lambda x: -x[1]['count']):
        category = categorize_merchant(merchant, data['descriptions'][0] if data['descriptions'] else '')
        
        # Get top patterns for this merchant
        sorted_patterns = sorted(data['patterns'].items(), key=lambda x: -x[1])
        top_patterns = sorted_patterns[:max_patterns_per_merchant]
        
        # Average amount
        avg_amount = sum(data['amounts']) / len(data['amounts']) if data['amounts'] else 0.0
        
        # Create one record per pattern
        for pattern_desc, count in top_patterns:
            # Find original description that normalizes to this pattern
            original_desc = None
            for desc in data['descriptions']:
                if normalize_transaction_string(desc) == pattern_desc:
                    original_desc = desc
                    break
            
            if original_desc:
                seed_records.append({
                    'merchant': merchant,
                    'description': original_desc,
                    'sample_amount': round(avg_amount, 2),
                    'category': category,
                    'pattern_count': count  # How many times this pattern appeared
                })
    
    # Create DataFrame
    seed_df = pd.DataFrame(seed_records)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save without pattern_count column (internal use only)
    seed_df_output = seed_df.drop(columns=['pattern_count'] if 'pattern_count' in seed_df.columns else [])
    seed_df_output.to_csv(output_path, index=False)
    
    logger.info(f"✅ Generated {len(seed_df)} merchant seed records")
    logger.info(f"✅ Saved to {output_file}")
    
    # Print summary
    print(f"\n📊 Expanded Merchant Seed Summary:")
    print(f"   Total records: {len(seed_df)}")
    print(f"   Unique merchants: {len(filtered_merchants)}")
    
    merchant_counts = seed_df.groupby('merchant').size()
    print(f"\n   Patterns per merchant:")
    for merchant, count in merchant_counts.head(15).items():
        print(f"     {merchant}: {count} patterns")
    
    category_counts = seed_df['category'].value_counts()
    print(f"\n   Category distribution:")
    for cat, count in category_counts.items():
        print(f"     {cat}: {count} patterns")
    
    return seed_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Expand merchants_seed.csv with multiple patterns')
    parser.add_argument('--input', type=str, default='data/raw_transactions.csv',
                       help='Input transactions CSV file')
    parser.add_argument('--output', type=str, default='data/merchants_seed.csv',
                       help='Output merchants seed CSV file')
    parser.add_argument('--min-occurrences', type=int, default=2,
                       help='Minimum occurrences to include merchant')
    parser.add_argument('--max-patterns', type=int, default=10,
                       help='Maximum patterns per merchant')
    
    args = parser.parse_args()
    
    print("🔧 Expanding merchants_seed.csv with multiple transaction patterns...")
    print("=" * 80)
    
    try:
        seed_df = expand_merchant_seed(
            transactions_file=args.input,
            output_file=args.output,
            min_occurrences=args.min_occurrences,
            max_patterns_per_merchant=args.max_patterns
        )
        
        print(f"\n✅ Successfully expanded to {len(seed_df)} merchant records!")
        print(f"✅ Next step: Rebuild FAISS index with: python utils/faiss_index_builder.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

