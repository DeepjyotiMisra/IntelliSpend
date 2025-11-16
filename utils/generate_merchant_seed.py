"""Generate merchants_seed.csv from raw_transactions.csv by extracting unique merchants"""

import sys
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import normalize_transaction_string
from config.settings import settings

logger = logging.getLogger(__name__)


def extract_merchant_from_description(description: str) -> str:
    """
    Extract merchant name from transaction description.
    Uses heuristics to identify merchant names.
    """
    desc_upper = description.upper()
    
    # Common merchant patterns
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
        'HP': ['HPCL', 'HP PETROL'],
        'INDIAN OIL': ['INDIAN OIL', 'INDIANOIL'],
        'AIRINDIA': ['AIRINDIA', 'AIR INDIA'],
        'IRCTC': ['IRCTC'],
        'MAKEMYTRIP': ['MAKEMYTRIP', 'MAKE MY TRIP'],
        'JIOMART': ['JIOMART', 'JIO MART'],
        'BIGBASKET': ['BIGBASKET', 'BIG BASKET'],
        'HDFC': ['HDFC BANK', 'HDFCBANK'],
        'ICICI': ['ICICI BANK', 'ICICIBANK'],
    }
    
    # Check patterns
    for merchant, patterns in merchant_patterns.items():
        for pattern in patterns:
            if pattern in desc_upper:
                return merchant
    
    # Try to extract from "TRANSFER TO X" or "X TXN" patterns
    transfer_match = re.search(r'TRANSFER TO ([A-Z\s]+)', desc_upper)
    if transfer_match:
        merchant = transfer_match.group(1).strip()
        # Clean up common suffixes
        merchant = re.sub(r'\s+\d+$', '', merchant)  # Remove trailing numbers
        return merchant
    
    # Try "X TXN" or "X DELHI/MUMBAI" patterns
    txn_match = re.search(r'^([A-Z\s]+?)\s+(?:DELHI|MUMBAI|IN|TXN)', desc_upper)
    if txn_match:
        merchant = txn_match.group(1).strip()
        return merchant
    
    # Try UPI pattern: UPI/xxx/MERCHANT
    upi_match = re.search(r'UPI/\d+/([A-Z]+)', desc_upper)
    if upi_match:
        return upi_match.group(1)
    
    return 'UNKNOWN'


def categorize_merchant(merchant: str, description: str) -> str:
    """
    Categorize merchant based on name and description.
    """
    merchant_upper = merchant.upper()
    desc_upper = description.upper()
    
    # Shopping
    if merchant_upper in ['AMAZON', 'FLIPKART', 'MYNTRA', 'NYKAA', 'JIOMART', 'BIGBASKET']:
        return 'Shopping'
    
    # Transport
    if merchant_upper in ['UBER', 'OLA'] or 'PETROL' in desc_upper or 'FUEL' in desc_upper:
        return 'Transport'
    if merchant_upper in ['SHELL', 'HP', 'INDIAN OIL', 'HPCL']:
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


def generate_merchant_seed_from_transactions(transactions_file: str = 'data/raw_transactions.csv',
                                            output_file: str = 'data/merchants_seed.csv',
                                            min_occurrences: int = 2) -> pd.DataFrame:
    """
    Generate merchants_seed.csv from raw_transactions.csv.
    
    Args:
        transactions_file: Path to raw transactions CSV
        output_file: Path to save merchants seed CSV
        min_occurrences: Minimum occurrences to include merchant
        
    Returns:
        DataFrame with merchant seed data
    """
    logger.info(f"Loading transactions from {transactions_file}...")
    df = pd.read_csv(transactions_file)
    
    # Extract merchants from descriptions
    merchant_data = defaultdict(lambda: {'descriptions': [], 'amounts': [], 'count': 0})
    
    for _, row in df.iterrows():
        desc = str(row.get('description', ''))
        amount = row.get('amount', 0.0)
        
        merchant = extract_merchant_from_description(desc)
        category = categorize_merchant(merchant, desc)
        
        if merchant != 'UNKNOWN':
            merchant_data[merchant]['descriptions'].append(desc)
            merchant_data[merchant]['amounts'].append(amount)
            merchant_data[merchant]['count'] += 1
            merchant_data[merchant]['category'] = category
    
    # Filter by minimum occurrences
    filtered_merchants = {k: v for k, v in merchant_data.items() 
                        if v['count'] >= min_occurrences}
    
    logger.info(f"Found {len(filtered_merchants)} unique merchants (min {min_occurrences} occurrences)")
    
    # Create seed data
    seed_records = []
    for merchant, data in sorted(filtered_merchants.items(), key=lambda x: -x[1]['count']):
        # Use most common description
        descriptions = data['descriptions']
        most_common_desc = max(set(descriptions), key=descriptions.count)
        
        # Average amount
        avg_amount = sum(data['amounts']) / len(data['amounts']) if data['amounts'] else 0.0
        
        seed_records.append({
            'merchant': merchant,
            'description': most_common_desc,
            'sample_amount': round(avg_amount, 2),
            'category': data.get('category', 'Other')
        })
    
    # Create DataFrame
    seed_df = pd.DataFrame(seed_records)
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Generated {len(seed_df)} merchant seed records")
    logger.info(f"✅ Saved to {output_file}")
    
    # Print summary
    print(f"\n📊 Generated Merchant Seed Summary:")
    print(f"   Total merchants: {len(seed_df)}")
    category_counts = seed_df['category'].value_counts()
    for cat, count in category_counts.items():
        print(f"   {cat}: {count}")
    
    return seed_df


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Generate merchants_seed.csv from transactions')
    parser.add_argument('--input', type=str, default='data/raw_transactions.csv',
                       help='Input transactions CSV file')
    parser.add_argument('--output', type=str, default='data/merchants_seed.csv',
                       help='Output merchants seed CSV file')
    parser.add_argument('--min-occurrences', type=int, default=2,
                       help='Minimum occurrences to include merchant')
    
    args = parser.parse_args()
    
    print("🔧 Generating merchants_seed.csv from raw_transactions.csv...")
    print("=" * 80)
    
    try:
        seed_df = generate_merchant_seed_from_transactions(
            transactions_file=args.input,
            output_file=args.output,
            min_occurrences=args.min_occurrences
        )
        
        print(f"\n✅ Successfully generated {len(seed_df)} merchant records!")
        print(f"✅ Next step: Build FAISS index with: python utils/faiss_index_builder.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

