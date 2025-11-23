"""CLI script to process user feedback and update merchant seed"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from config.settings import settings
from agents.feedback_agent import process_feedback
from utils.feedback_processor import (
    load_feedback,
    get_pending_feedback,
    update_merchant_seed_from_feedback,
    get_feedback_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def add_feedback_interactive():
    """Interactive mode to add feedback"""
    print("📝 Add Feedback")
    print("=" * 60)
    print()
    
    transaction_desc = input("Transaction description: ").strip()
    original_merchant = input("Original merchant: ").strip()
    original_category = input("Original category: ").strip()
    corrected_merchant = input("Corrected merchant: ").strip()
    corrected_category = input("Corrected category: ").strip()
    
    amount = input("Amount (optional, press Enter to skip): ").strip()
    amount = float(amount) if amount else None
    
    date = input("Date (optional, press Enter to skip): ").strip()
    date = date if date else None
    
    confidence = input("Original confidence score (optional, press Enter to skip): ").strip()
    confidence = float(confidence) if confidence else None
    
    print()
    print("Processing feedback with LLM agent...")
    
    response = process_feedback(
        transaction_description=transaction_desc,
        original_merchant=original_merchant,
        original_category=original_category,
        corrected_merchant=corrected_merchant,
        corrected_category=corrected_category,
        amount=amount,
        date=date,
        confidence_score=confidence
    )
    
    print()
    print("🤖 Feedback Agent Response:")
    print("-" * 60)
    print(response)
    print()


def add_feedback_from_csv(csv_file: str):
    """Add feedback from CSV file"""
    print(f"📝 Adding feedback from {csv_file}")
    print("=" * 60)
    print()
    
    try:
        df = pd.read_csv(csv_file)
        
        required_cols = ['transaction_description', 'original_merchant', 'original_category',
                        'corrected_merchant', 'corrected_category']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ Error: Missing required columns: {missing}")
            return
        
        print(f"Found {len(df)} feedback records")
        print()
        
        for idx, row in df.iterrows():
            print(f"Processing feedback {idx+1}/{len(df)}...")
            
            response = process_feedback(
                transaction_description=row['transaction_description'],
                original_merchant=row['original_merchant'],
                original_category=row['original_category'],
                corrected_merchant=row['corrected_merchant'],
                corrected_category=row['corrected_category'],
                amount=row.get('amount'),
                date=row.get('date'),
                confidence_score=row.get('confidence_score')
            )
            
            print(f"✅ Feedback {idx+1} processed")
        
        print()
        print("✅ All feedback processed!")
        
    except Exception as e:
        logger.error(f"Error processing CSV feedback: {e}")
        print(f"❌ Error: {e}")


def show_feedback_summary():
    """Show feedback statistics"""
    print("📊 Feedback Summary")
    print("=" * 60)
    print()
    
    summary = get_feedback_summary()
    
    print(f"Total Feedback: {summary.get('total', 0)}")
    print(f"Processed: {summary.get('processed', 0)}")
    print(f"Pending: {summary.get('pending', 0)}")
    print()
    
    if summary.get('most_corrected_merchants'):
        print("Most Corrected Merchants:")
        for merchant, count in list(summary['most_corrected_merchants'].items())[:5]:
            print(f"  {merchant}: {count}")
        print()
    
    if summary.get('most_corrected_categories'):
        print("Most Corrected Categories:")
        for category, count in list(summary['most_corrected_categories'].items())[:5]:
            print(f"  {category}: {count}")
        print()


def apply_feedback(rebuild_index: bool = True):
    """Apply pending feedback to merchant seed"""
    print("🔄 Applying Feedback to Merchant Seed")
    print("=" * 60)
    print()
    
    result = update_merchant_seed_from_feedback(rebuild_index=rebuild_index)
    
    if result.get('success'):
        print(f"✅ {result.get('message')}")
        print(f"   New patterns: {result.get('new_patterns', 0)}")
        print(f"   Updated merchants: {result.get('updated_merchants', 0)}")
        if result.get('merchants'):
            print(f"   Merchants: {', '.join(result['merchants'][:10])}")
            if len(result['merchants']) > 10:
                print(f"   ... and {len(result['merchants']) - 10} more")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process feedback for IntelliSpend')
    parser.add_argument('--add', action='store_true',
                       help='Add feedback interactively')
    parser.add_argument('--add-csv', type=str,
                       help='Add feedback from CSV file')
    parser.add_argument('--summary', action='store_true',
                       help='Show feedback summary')
    parser.add_argument('--apply', action='store_true',
                       help='Apply pending feedback to merchant seed')
    parser.add_argument('--no-rebuild', action='store_true',
                       help='Skip FAISS index rebuild when applying feedback')
    
    args = parser.parse_args()
    
    if args.add:
        add_feedback_interactive()
    elif args.add_csv:
        add_feedback_from_csv(args.add_csv)
    elif args.summary:
        show_feedback_summary()
    elif args.apply:
        apply_feedback(rebuild_index=not args.no_rebuild)
    else:
        parser.print_help()

