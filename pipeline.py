"""Main pipeline for processing raw transactions through IntelliSpend agents"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm

from utils.data_utils import load_transactions, normalize_transaction_string
from agents.tools import normalize_transaction, retrieve_similar_merchants
from config.settings import settings

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


def process_single_transaction(description: str, amount: float = None, date: str = None) -> Dict[str, Any]:
    """
    Process a single transaction through the pipeline.
    
    Args:
        description: Transaction description
        amount: Transaction amount (optional)
        date: Transaction date (optional)
        
    Returns:
        Dict with processed transaction data
    """
    try:
        # Step 1: Preprocess
        preprocess_result = normalize_transaction(description)
        normalized = preprocess_result.get('normalized', description)
        payment_mode = preprocess_result.get('payment_mode', 'UNKNOWN')
        
        # Step 2: Retrieve similar merchants
        # Use a lower threshold for initial matching, then filter by confidence
        retrieve_result = retrieve_similar_merchants(normalized, top_k=3, min_score=None)
        
        # Extract best match
        best_match = retrieve_result.get('best_match')
        all_matches = retrieve_result.get('results', [])
        
        # Use best match if confidence is above threshold, otherwise use best available
        if best_match and best_match.get('score', 0) >= settings.LOCAL_MATCH_THRESHOLD:
            merchant = best_match.get('merchant', 'UNKNOWN')
            category = best_match.get('category', 'Other')
            confidence = best_match.get('score', 0.0)
            match_quality = 'high'
        elif best_match:
            # Use best match even if below threshold, but mark as low confidence
            merchant = best_match.get('merchant', 'UNKNOWN')
            category = best_match.get('category', 'Other')
            confidence = best_match.get('score', 0.0)
            match_quality = 'low'
        else:
            merchant = 'UNKNOWN'
            category = 'Other'
            confidence = 0.0
            match_quality = 'none'
        
        result = {
            'original_description': description,
            'normalized_description': normalized,
            'payment_mode': payment_mode,
            'amount': amount,
            'date': date,
            'merchant': merchant,
            'category': category,
            'confidence_score': confidence,
            'match_quality': match_quality,
            'num_matches': retrieve_result.get('num_results', 0),
            'retrieval_source': 'faiss' if retrieve_result.get('success') else 'none',
            'processing_status': 'success',
            'top_matches': all_matches[:3] if all_matches else []
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing transaction '{description}': {e}")
        return {
            'original_description': description,
            'normalized_description': description,
            'payment_mode': 'UNKNOWN',
            'amount': amount,
            'date': date,
            'merchant': 'UNKNOWN',
            'category': 'Other',
            'confidence_score': 0.0,
            'num_matches': 0,
            'retrieval_source': 'error',
            'processing_status': 'error',
            'error_message': str(e)
        }


def process_batch_transactions(descriptions: List[str], 
                               amounts: List[float] = None,
                               dates: List[str] = None,
                               batch_size: int = 100) -> List[Dict[str, Any]]:
    """
    Process multiple transactions in batches.
    
    Args:
        descriptions: List of transaction descriptions
        amounts: List of amounts (optional)
        dates: List of dates (optional)
        batch_size: Number of transactions to process per batch
        
    Returns:
        List of processed transaction results
    """
    results = []
    total = len(descriptions)
    
    # Ensure amounts and dates lists match length
    if amounts is None:
        amounts = [None] * total
    if dates is None:
        dates = [None] * total
    
    logger.info(f"Processing {total} transactions in batches of {batch_size}...")
    
    for i in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch_descriptions = descriptions[i:i+batch_size]
        batch_amounts = amounts[i:i+batch_size]
        batch_dates = dates[i:i+batch_size]
        
        batch_results = []
        for desc, amount, date in zip(batch_descriptions, batch_amounts, batch_dates):
            result = process_single_transaction(desc, amount, date)
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results


def process_transactions_file(input_file: str = 'data/raw_transactions.csv',
                             output_file: str = 'output/categorized_transactions.csv',
                             description_col: str = 'description',
                             amount_col: str = 'amount',
                             date_col: str = 'date',
                             batch_size: int = 100) -> pd.DataFrame:
    """
    Process a CSV file of transactions through the IntelliSpend pipeline.
    
    Args:
        input_file: Path to input CSV file
        description_col: Name of description column
        amount_col: Name of amount column
        date_col: Name of date column
        output_file: Path to save output CSV
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with categorized transactions
    """
    logger.info(f"Loading transactions from {input_file}...")
    
    # Load transactions
    df = load_transactions(
        input_file,
        description_col=description_col,
        amount_col=amount_col,
        date_col=date_col
    )
    
    total = len(df)
    logger.info(f"Loaded {total} transactions")
    
    # Extract columns
    descriptions = df[description_col].tolist()
    amounts = df[amount_col].tolist() if amount_col in df.columns else [None] * total
    dates = df[date_col].tolist() if date_col in df.columns else [None] * total
    
    # Process transactions
    start_time = time.time()
    results = process_batch_transactions(descriptions, amounts, dates, batch_size=batch_size)
    processing_time = time.time() - start_time
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original data
    output_df = df.copy()
    
    # Prepare merge columns (exclude nested structures)
    merge_cols = ['original_description', 'merchant', 'category', 'confidence_score', 
                  'payment_mode', 'num_matches', 'retrieval_source', 'processing_status', 'match_quality']
    available_cols = [col for col in merge_cols if col in results_df.columns]
    
    output_df = output_df.merge(
        results_df[available_cols],
        left_on=description_col,
        right_on='original_description',
        how='left'
    )
    
    # Drop duplicate column
    if 'original_description' in output_df.columns:
        output_df = output_df.drop(columns=['original_description'])
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Processed {total} transactions in {processing_time:.2f} seconds")
    logger.info(f"✅ Average processing time: {processing_time/total*1000:.2f} ms per transaction")
    logger.info(f"✅ Results saved to {output_file}")
    
    # Print statistics
    print_statistics(output_df)
    
    return output_df


def print_statistics(df: pd.DataFrame):
    """Print processing statistics"""
    total = len(df)
    
    # Count by status
    if 'processing_status' in df.columns:
        success_count = (df['processing_status'] == 'success').sum()
        error_count = (df['processing_status'] == 'error').sum()
        print(f"\n📊 Processing Statistics:")
        print(f"   Total transactions: {total}")
        print(f"   Successful: {success_count} ({success_count/total*100:.1f}%)")
        print(f"   Errors: {error_count} ({error_count/total*100:.1f}%)")
    
    # Count by category
    if 'category' in df.columns:
        print(f"\n📊 Category Distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.head(10).items():
            print(f"   {category}: {count} ({count/total*100:.1f}%)")
    
    # Confidence statistics
    if 'confidence_score' in df.columns:
        avg_confidence = df['confidence_score'].mean()
        high_confidence = (df['confidence_score'] >= settings.CONFIDENCE_HIGH_THRESHOLD).sum()
        medium_confidence = ((df['confidence_score'] >= settings.CONFIDENCE_MEDIUM_THRESHOLD) & 
                            (df['confidence_score'] < settings.CONFIDENCE_HIGH_THRESHOLD)).sum()
        low_confidence = (df['confidence_score'] < settings.CONFIDENCE_MEDIUM_THRESHOLD).sum()
        
        print(f"\n📊 Confidence Statistics:")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   High confidence (≥{settings.CONFIDENCE_HIGH_THRESHOLD}): {high_confidence} ({high_confidence/total*100:.1f}%)")
        print(f"   Medium confidence ({settings.CONFIDENCE_MEDIUM_THRESHOLD}-{settings.CONFIDENCE_HIGH_THRESHOLD}): {medium_confidence} ({medium_confidence/total*100:.1f}%)")
        print(f"   Low confidence (<{settings.CONFIDENCE_MEDIUM_THRESHOLD}): {low_confidence} ({low_confidence/total*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process transactions through IntelliSpend pipeline')
    parser.add_argument('--input', type=str, default='data/raw_transactions.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='output/categorized_transactions.csv',
                       help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--description-col', type=str, default='description',
                       help='Name of description column')
    parser.add_argument('--amount-col', type=str, default='amount',
                       help='Name of amount column')
    parser.add_argument('--date-col', type=str, default='date',
                       help='Name of date column')
    
    args = parser.parse_args()
    
    print("🚀 IntelliSpend Transaction Processing Pipeline")
    print("=" * 80)
    print()
    
    try:
        # Process transactions
        results_df = process_transactions_file(
            input_file=args.input,
            output_file=args.output,
            description_col=args.description_col,
            amount_col=args.amount_col,
            date_col=args.date_col,
            batch_size=args.batch_size
        )
        
        print("\n✅ Processing complete!")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("💡 Please check:")
        print("   1. Input file exists and is readable")
        print("   2. FAISS index is built (run: python utils/faiss_index_builder.py)")
        print("   3. All dependencies are installed")
        raise

