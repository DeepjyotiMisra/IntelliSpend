"""Main pipeline for processing raw transactions through IntelliSpend agents"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.data_utils import load_transactions, normalize_transaction_string
from agents.tools import normalize_transaction, retrieve_similar_merchants, store_feedback
from config.settings import settings
import re

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


def parse_classifier_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the Classifier Agent's response to extract structured information.
    
    Args:
        response_text: Raw response text from the agent
        
    Returns:
        Dict with merchant, category, confidence, and reasoning
    """
    result = {
        'merchant': 'UNKNOWN',
        'category': 'Other',
        'confidence': 'medium',
        'reasoning': response_text[:200]  # First 200 chars as reasoning
    }
    
    response_lower = response_text.lower()
    
    # Try to extract merchant
    merchant_patterns = [
        r'(?:merchant|identified\s+merchant)["\']?\s*:?\s*["\']?([^"\'\n,]+)',
        r'(?:merchant|identified\s+merchant)["\']?\s*:?\s*([A-Z][A-Z\s&]+)',
        r'\*\*IDENTIFIED\s+MERCHANT:\*\*\s*([A-Z][A-Z\s&]+)',
        r'merchant["\']?\s*:?\s*\*\*([^*]+)\*\*',
        r'merchant["\']?\s*:?\s*([A-Z][A-Z\s&]+)',
    ]
    for pattern in merchant_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            merchant = match.group(1).strip().strip('"').strip("'").strip('*').strip('-').strip()
            # Remove markdown formatting
            merchant = re.sub(r'\*\*', '', merchant)  # Remove bold
            merchant = re.sub(r'[#*-]', '', merchant)  # Remove bullets, dashes
            merchant = merchant.strip()
            if len(merchant) < 50 and merchant.upper() != 'UNKNOWN' and not merchant.startswith('IDENTIFIED'):
                result['merchant'] = merchant.upper()
                break
    
    # Try to extract category
    # First, look for explicit category assignments (e.g., "Category: Food & Dining")
    category_patterns = [
        r'category["\']?\s*:?\s*["\']?([^"\'\n,]+)',
        r'category["\']?\s*:?\s*([A-Z][A-Z\s&]+)',
        r'\*\*category\*\*["\']?\s*:?\s*["\']?([^"\'\n,]+)',  # Markdown format
    ]
    
    for pattern in category_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            category_text = match.group(1).strip().strip('*').strip('"').strip("'")
            # Check if it's a valid category
            categories = ['Shopping', 'Bills & Payments', 'Food & Dining', 'Banking', 
                          'Transport', 'Entertainment', 'Other']
            for cat in categories:
                if cat.lower() in category_text.lower():
                    result['category'] = cat
                    break
            if result['category'] != 'Other':
                break
    
    # Fallback: search for category names in context (not just anywhere)
    if result['category'] == 'Other':
        categories = ['Food & Dining', 'Shopping', 'Bills & Payments', 'Banking', 
                      'Transport', 'Entertainment', 'Other']
        # Look for category in structured sections (after "Category:" or in reasoning)
        for cat in categories:
            # Check if category appears in a structured context
            pattern = rf'(?:category|classified|belongs? to|falls? under)[\s:]+{re.escape(cat)}'
            if re.search(pattern, response_lower, re.IGNORECASE):
                result['category'] = cat
                break
    
    # Try to extract confidence
    if 'high' in response_lower and 'confidence' in response_lower:
        result['confidence'] = 'high'
    elif 'low' in response_lower and 'confidence' in response_lower:
        result['confidence'] = 'low'
    elif 'medium' in response_lower and 'confidence' in response_lower:
        result['confidence'] = 'medium'
    
    return result


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
        
        # Step 3: Classification
        # Automatically use Classifier Agent for low-confidence matches
        confidence_score = best_match.get('score', 0.0) if best_match else 0.0
        should_use_agent = confidence_score < settings.CLASSIFIER_AGENT_THRESHOLD
        
        if should_use_agent:
            # Use Classifier Agent for low-confidence matches
            logger.info(f"Using Classifier Agent for low-confidence match (score: {confidence_score:.3f})")
            # Note: Progress logging happens at batch level, not per transaction to avoid spam
            try:
                from agents.classifier_agent import classify_transaction as agent_classify
                agent_response = agent_classify(
                    description=description,
                    amount=amount,
                    payment_mode=payment_mode,
                    retrieved_merchants=all_matches
                )
                
                # Parse agent response
                agent_result = parse_classifier_response(agent_response)
                merchant = agent_result.get('merchant', 'UNKNOWN')
                category = agent_result.get('category', 'Other')
                agent_confidence = agent_result.get('confidence', 'medium')
                
                # Map confidence text to score
                confidence_map = {'high': 0.85, 'medium': 0.70, 'low': 0.50}
                confidence = confidence_map.get(agent_confidence, confidence_score)
                match_quality = 'agent_llm'
                classification_source = 'llm'
                
            except Exception as e:
                logger.warning(f"Classifier Agent failed, falling back to direct classification: {e}")
                # Fallback to direct classification
                if best_match:
                    merchant = best_match.get('merchant', 'UNKNOWN')
                    category = best_match.get('category', 'Other')
                    confidence = best_match.get('score', 0.0)
                    match_quality = 'low'
                else:
                    merchant = 'UNKNOWN'
                    category = 'Other'
                    confidence = 0.0
                    match_quality = 'none'
                classification_source = 'direct_fallback'
        else:
            # Direct classification (fast path) - high confidence
            if best_match and confidence_score >= settings.LOCAL_MATCH_THRESHOLD:
                merchant = best_match.get('merchant', 'UNKNOWN')
                category = best_match.get('category', 'Other')
                confidence = confidence_score
                match_quality = 'high'
            elif best_match:
                merchant = best_match.get('merchant', 'UNKNOWN')
                category = best_match.get('category', 'Other')
                confidence = confidence_score
                match_quality = 'low'
            else:
                merchant = 'UNKNOWN'
                category = 'Other'
                confidence = 0.0
                match_quality = 'none'
            classification_source = 'direct'
        
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
            'classification_source': classification_source,
            'processing_status': 'success',
            'top_matches': all_matches[:3] if all_matches else []
        }
        
        # Track custom category usage
        try:
            from utils.custom_categories import is_custom_category, increment_category_usage
            if is_custom_category(category):
                increment_category_usage(category)
        except Exception as e:
            logger.debug(f"Could not track custom category usage: {e}")
        
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


def _process_with_retrieved_matches(description: str, normalized: str, amount: float = None, 
                                     date: str = None, best_match: Dict[str, Any] = None,
                                     all_matches: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a transaction using pre-retrieved matches (optimized path for batch processing).
    
    Args:
        description: Original transaction description
        normalized: Normalized transaction description
        amount: Transaction amount (optional)
        date: Transaction date (optional)
        best_match: Pre-retrieved best match
        all_matches: Pre-retrieved all matches
        
    Returns:
        Dict with processed transaction data
    """
    try:
        # Extract payment mode
        from utils.data_utils import extract_payment_mode
        payment_mode = extract_payment_mode(description)
        
        # Use pre-retrieved matches
        if not all_matches:
            all_matches = []
        if not best_match and all_matches:
            best_match = all_matches[0]
        
        # Step 3: Classification
        # Automatically use Classifier Agent for low-confidence matches
        confidence_score = best_match.get('score', 0.0) if best_match else 0.0
        should_use_agent = confidence_score < settings.CLASSIFIER_AGENT_THRESHOLD
        
        if should_use_agent:
            # Use Classifier Agent for low-confidence matches
            logger.debug(f"Using Classifier Agent for low-confidence match (score: {confidence_score:.3f})")
            try:
                from agents.classifier_agent import classify_transaction as agent_classify
                agent_response = agent_classify(
                    description=description,
                    amount=amount,
                    payment_mode=payment_mode,
                    retrieved_merchants=all_matches
                )
                
                # Parse agent response
                agent_result = parse_classifier_response(agent_response)
                merchant = agent_result.get('merchant', 'UNKNOWN')
                category = agent_result.get('category', 'Other')
                agent_confidence = agent_result.get('confidence', 'medium')
                
                # Map confidence text to score
                confidence_map = {'high': 0.85, 'medium': 0.70, 'low': 0.50}
                confidence = confidence_map.get(agent_confidence, confidence_score)
                match_quality = 'agent_llm'
                classification_source = 'llm'
                
            except Exception as e:
                logger.warning(f"Classifier Agent failed, falling back to direct classification: {e}")
                # Fallback to direct classification
                if best_match:
                    merchant = best_match.get('merchant', 'UNKNOWN')
                    category = best_match.get('category', 'Other')
                    confidence = best_match.get('score', 0.0)
                    match_quality = 'low'
                else:
                    merchant = 'UNKNOWN'
                    category = 'Other'
                    confidence = 0.0
                    match_quality = 'none'
                classification_source = 'direct_fallback'
        else:
            # Direct classification (fast path) - high confidence
            if best_match and confidence_score >= settings.LOCAL_MATCH_THRESHOLD:
                merchant = best_match.get('merchant', 'UNKNOWN')
                category = best_match.get('category', 'Other')
                confidence = confidence_score
                match_quality = 'high'
            elif best_match:
                merchant = best_match.get('merchant', 'UNKNOWN')
                category = best_match.get('category', 'Other')
                confidence = confidence_score
                match_quality = 'low'
            else:
                merchant = 'UNKNOWN'
                category = 'Other'
                confidence = 0.0
                match_quality = 'none'
            classification_source = 'direct'
        
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
            'num_matches': len(all_matches),
            'retrieval_source': 'faiss' if best_match else 'none',
            'classification_source': classification_source,
            'processing_status': 'success',
            'top_matches': all_matches[:3] if all_matches else []
        }
        
        # Track custom category usage
        try:
            from utils.custom_categories import is_custom_category, increment_category_usage
            if is_custom_category(category):
                increment_category_usage(category)
        except Exception as e:
            logger.debug(f"Could not track custom category usage: {e}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing transaction '{description}': {e}")
        return {
            'original_description': description,
            'normalized_description': normalized if 'normalized' in locals() else description,
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
                               batch_size: int = 100,
                               progress_callback: callable = None) -> List[Dict[str, Any]]:
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
    print(f"📊 Processing {total} transactions...")
    print(f"   Batch size: {batch_size}")
    print()
    
    processed_count = 0
    llm_count = 0
    direct_count = 0
    start_time = time.time()
    
    # Determine progress interval based on total (more frequent for larger sets)
    if total <= 10:
        progress_interval = 1  # Every transaction for small sets
    elif total <= 50:
        progress_interval = 5  # Every 5 transactions
    elif total <= 200:
        progress_interval = 10  # Every 10 transactions
    else:
        progress_interval = 20  # Every 20 transactions for large sets
    
    # OPTIMIZATION: Batch process embeddings for better performance
    # Update progress: Initializing retriever (10-15%)
    if progress_callback:
        progress_callback(0, total, 10.0, 0, 0, "")
    
    from utils.faiss_retriever import get_retriever
    from utils.data_utils import normalize_transaction_string
    
    # Initialize retriever (this may load FAISS index and embedding model)
    # Add granular progress updates during initialization
    logger.info("Initializing FAISS retriever and embedding model...")
    if progress_callback:
        progress_callback(0, total, 11.0, 0, 0, "🔄 Loading embedding model...")
    
    # Pre-load embedding service with progress update
    # This is the slowest part - loading the SentenceTransformer model
    from utils.embedding_service import get_embedding_service
    embedding_service = get_embedding_service()  # This loads the model synchronously
    
    if progress_callback:
        progress_callback(0, total, 12.5, 0, 0, "🔄 Loading FAISS index...")
    
    retriever = get_retriever(top_k=3)
    
    # Update progress: Retriever initialized (15%)
    if progress_callback:
        progress_callback(0, total, 15.0, 0, 0, "✅ Initialization complete")
    
    # Pre-check if index is loaded
    index_loaded = retriever.is_index_loaded()
    if not index_loaded:
        logger.warning("FAISS index not loaded, processing will be slower")
    
    for i in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch_descriptions = descriptions[i:i+batch_size]
        batch_amounts = amounts[i:i+batch_size]
        batch_dates = dates[i:i+batch_size]
        actual_batch_size = len(batch_descriptions)
        
        # OPTIMIZATION: Batch normalize all descriptions at once
        batch_normalized = [normalize_transaction_string(desc) for desc in batch_descriptions]
        
        # OPTIMIZATION: Batch retrieve similar merchants (generates embeddings in batch)
        batch_retrieve_results = []
        if index_loaded:
            try:
                # Use batch_retrieve for efficient embedding generation
                # This generates all embeddings in one call, then searches FAISS in batch
                logger.debug(f"Batch retrieving for {actual_batch_size} transactions...")
                # Update progress: Starting batch retrieval (helps show activity for large files)
                if progress_callback and i == 0:  # First batch only
                    progress_callback(0, total, 16.0, 0, 0, "")
                batch_retrieve_results = retriever.batch_retrieve(batch_normalized, top_k=3, min_score=None)
                logger.debug(f"Batch retrieve completed: {len(batch_retrieve_results)} results")
                # Update progress: Batch retrieval complete
                if progress_callback and i == 0:  # First batch only
                    progress_callback(0, total, 17.0, 0, 0, "")
            except Exception as e:
                logger.warning(f"Batch retrieve failed, falling back to individual: {e}")
                batch_retrieve_results = []
        
        # OPTIMIZATION: Separate transactions that need LLM vs direct classification
        # Process direct classifications first (fast), then LLM in parallel (slower)
        direct_results = {}  # index -> result
        llm_tasks = []  # List of (index, desc, normalized, amount, date, best_match, all_matches)
        
        for j in range(actual_batch_size):
            desc = batch_descriptions[j]
            amount = batch_amounts[j] if batch_amounts else None
            date = batch_dates[j] if batch_dates else None
            normalized = batch_normalized[j]
            
            # Use pre-retrieved results if available
            if batch_retrieve_results and j < len(batch_retrieve_results):
                retrieved_matches = batch_retrieve_results[j]
                best_match = retrieved_matches[0] if retrieved_matches else None
                all_matches = retrieved_matches
                confidence_score = best_match.get('score', 0.0) if best_match else 0.0
                
                # Check if LLM is needed
                needs_llm = (confidence_score < settings.CLASSIFIER_AGENT_THRESHOLD)
                
                if needs_llm:
                    # Queue for parallel LLM processing
                    llm_tasks.append((j, desc, normalized, amount, date, best_match, all_matches))
                else:
                    # Process directly (fast path - no LLM needed)
                    result = _process_with_retrieved_matches(
                        desc, normalized, amount, date, best_match, all_matches
                    )
                    direct_results[j] = result
            else:
                # Fallback to individual processing (SLOW PATH - should rarely happen)
                result = process_single_transaction(desc, amount, date)
                direct_results[j] = result
        
        # OPTIMIZATION: Process LLM calls in parallel
        llm_results = {}
        if llm_tasks:
            logger.debug(f"Processing {len(llm_tasks)} transactions with LLM in parallel...")
            
            def process_llm_task(task_data):
                """Process a single LLM task"""
                j, desc, normalized, amount, date, best_match, all_matches = task_data
                try:
                    from utils.data_utils import extract_payment_mode
                    payment_mode = extract_payment_mode(desc)
                    
                    from agents.classifier_agent import classify_transaction as agent_classify
                    agent_response = agent_classify(
                        description=desc,
                        amount=amount,
                        payment_mode=payment_mode,
                        retrieved_merchants=all_matches
                    )
                    
                    # Parse agent response
                    agent_result = parse_classifier_response(agent_response)
                    merchant = agent_result.get('merchant', 'UNKNOWN')
                    category = agent_result.get('category', 'Other')
                    agent_confidence = agent_result.get('confidence', 'medium')
                    
                    # Map confidence text to score
                    confidence_map = {'high': 0.85, 'medium': 0.70, 'low': 0.50}
                    confidence_score = best_match.get('score', 0.0) if best_match else 0.0
                    confidence = confidence_map.get(agent_confidence, confidence_score)
                    
                    return j, {
                        'original_description': desc,
                        'normalized_description': normalized,
                        'payment_mode': payment_mode,
                        'amount': amount,
                        'date': date,
                        'merchant': merchant,
                        'category': category,
                        'confidence_score': confidence,
                        'match_quality': 'agent_llm',
                        'num_matches': len(all_matches),
                        'retrieval_source': 'faiss' if best_match else 'none',
                        'classification_source': 'llm',
                        'processing_status': 'success',
                        'top_matches': all_matches[:3] if all_matches else []
                    }
                except Exception as e:
                    logger.warning(f"LLM processing failed for transaction {j}: {e}")
                    # Fallback to direct classification
                    from utils.data_utils import extract_payment_mode
                    payment_mode = extract_payment_mode(desc)
                    
                    if best_match:
                        merchant = best_match.get('merchant', 'UNKNOWN')
                        category = best_match.get('category', 'Other')
                        confidence = best_match.get('score', 0.0)
                    else:
                        merchant = 'UNKNOWN'
                        category = 'Other'
                        confidence = 0.0
                    
                    return j, {
                        'original_description': desc,
                        'normalized_description': normalized,
                        'payment_mode': payment_mode,
                        'amount': amount,
                        'date': date,
                        'merchant': merchant,
                        'category': category,
                        'confidence_score': confidence,
                        'match_quality': 'low',
                        'num_matches': len(all_matches),
                        'retrieval_source': 'faiss' if best_match else 'none',
                        'classification_source': 'direct_fallback',
                        'processing_status': 'success',
                        'top_matches': all_matches[:3] if all_matches else []
                    }
            
            # Process LLM tasks in parallel (max 3 concurrent LLM calls to avoid rate limits and segfaults)
            # Reduced from 5 to 3 to avoid multiprocessing conflicts that can cause segfaults
            max_workers = min(3, len(llm_tasks))
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {executor.submit(process_llm_task, task): task[0] for task in llm_tasks}
                    
                    for future in as_completed(future_to_index):
                        try:
                            j, result = future.result(timeout=120)  # Add timeout to prevent hanging
                            llm_results[j] = result
                        except Exception as e:
                            logger.error(f"Error in parallel LLM processing: {e}")
                            j = future_to_index[future]
                            # Create error result
                            llm_results[j] = {
                                'original_description': batch_descriptions[j],
                                'normalized_description': batch_normalized[j],
                                'payment_mode': 'UNKNOWN',
                                'amount': batch_amounts[j] if batch_amounts else None,
                                'date': batch_dates[j] if batch_dates else None,
                                'merchant': 'UNKNOWN',
                                'category': 'Other',
                                'confidence_score': 0.0,
                                'num_matches': 0,
                                'retrieval_source': 'error',
                                'classification_source': 'error',
                                'processing_status': 'error',
                                'error_message': str(e)
                            }
            except Exception as e:
                logger.error(f"Critical error in parallel LLM processing: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback: process sequentially if parallel processing fails
                logger.warning("Falling back to sequential LLM processing due to error")
                for task in llm_tasks:
                    try:
                        j, result = process_llm_task(task)
                        llm_results[j] = result
                    except Exception as e2:
                        logger.error(f"Error in sequential LLM processing: {e2}")
                        j = task[0]
                        llm_results[j] = {
                            'original_description': batch_descriptions[j],
                            'normalized_description': batch_normalized[j],
                            'payment_mode': 'UNKNOWN',
                            'amount': batch_amounts[j] if batch_amounts else None,
                            'date': batch_dates[j] if batch_dates else None,
                            'merchant': 'UNKNOWN',
                            'category': 'Other',
                            'confidence_score': 0.0,
                            'num_matches': 0,
                            'retrieval_source': 'error',
                            'classification_source': 'error',
                            'processing_status': 'error',
                            'error_message': str(e2)
                        }
        
        # Combine direct and LLM results in correct order
        batch_results = []
        for j in range(actual_batch_size):
            if j in direct_results:
                result = direct_results[j]
            elif j in llm_results:
                result = llm_results[j]
            else:
                # Fallback (shouldn't happen)
                result = process_single_transaction(
                    batch_descriptions[j],
                    batch_amounts[j] if batch_amounts else None,
                    batch_dates[j] if batch_dates else None
                )
            
            batch_results.append(result)
            processed_count += 1
            
            # Track classification sources
            if result.get('classification_source') == 'llm':
                llm_count += 1
            elif result.get('classification_source') == 'direct':
                direct_count += 1
            
            # Call progress callback if provided (for UI updates)
            if progress_callback:
                progress_pct = (processed_count / total) * 100
                elapsed = time.time() - start_time
                if processed_count > 0:
                    avg_time_per_txn = elapsed / processed_count
                    remaining = total - processed_count
                    eta_seconds = avg_time_per_txn * remaining
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f" (ETA: {eta_min}m {eta_sec}s)" if remaining > 0 else ""
                else:
                    eta_str = ""
                progress_callback(processed_count, total, progress_pct, llm_count, direct_count, eta_str)
            
            # Log progress at intervals or at milestones
            if processed_count % progress_interval == 0 or processed_count == total:
                progress_pct = (processed_count / total) * 100
                elapsed = time.time() - start_time
                if processed_count > 0:
                    avg_time_per_txn = elapsed / processed_count
                    remaining = total - processed_count
                    eta_seconds = avg_time_per_txn * remaining
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f" (ETA: {eta_min}m {eta_sec}s)" if remaining > 0 else ""
                else:
                    eta_str = ""
                
                print(f"   ✅ Processed: {processed_count}/{total} ({progress_pct:.1f}%) | "
                      f"LLM: {llm_count} | Direct: {direct_count}{eta_str}")
        
        results.extend(batch_results)
    
    total_time = time.time() - start_time
    print()
    print(f"✅ Completed: {processed_count}/{total} transactions in {total_time:.1f}s")
    print(f"   LLM Classifications: {llm_count} ({llm_count/processed_count*100:.1f}%)")
    print(f"   Direct Classifications: {direct_count} ({direct_count/processed_count*100:.1f}%)")
    if processed_count > 0:
        print(f"   Average time: {total_time/processed_count:.2f}s per transaction")
    print()
    
    return results


def process_transactions_file(input_file: str = 'data/raw_transactions.csv',
                             output_file: str = 'output/categorized_transactions.csv',
                             description_col: str = 'description',
                             amount_col: str = 'amount',
                             date_col: str = 'date',
                             batch_size: int = 100,
                             progress_callback: callable = None) -> pd.DataFrame:
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
    
    # Update progress: Loading file (5%)
    if progress_callback:
        progress_callback(0, 1, 5.0, 0, 0, "")
    
    # Load transactions
    df = load_transactions(
        input_file,
        description_col=description_col,
        amount_col=amount_col,
        date_col=date_col
    )
    
    total = len(df)
    logger.info(f"Loaded {total} transactions")
    
    # Update progress: File loaded, initializing (10%)
    if progress_callback:
        progress_callback(0, total, 10.0, 0, 0, "")
    
    # Extract columns
    # Note: load_transactions normalizes column names to lowercase ('description', 'amount', 'date')
    # so we use the standard names regardless of what was passed in
    descriptions = df['description'].tolist()
    amounts = df['amount'].tolist() if 'amount' in df.columns else [None] * total
    dates = df['date'].tolist() if 'date' in df.columns else [None] * total
    
    # Process transactions
    start_time = time.time()
    results = process_batch_transactions(descriptions, amounts, dates, batch_size=batch_size, progress_callback=progress_callback)
    processing_time = time.time() - start_time
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original data
    output_df = df.copy()
    
    # Prepare merge columns (exclude nested structures)
    merge_cols = ['original_description', 'merchant', 'category', 'confidence_score', 
                  'payment_mode', 'num_matches', 'retrieval_source', 'processing_status', 
                  'match_quality', 'classification_source']
    available_cols = [col for col in merge_cols if col in results_df.columns]
    
    # Ensure 'original_description' is always included for the merge
    if 'original_description' not in available_cols and 'original_description' in results_df.columns:
        available_cols.append('original_description')
    
    # Note: load_transactions normalizes column names to lowercase ('description', 'amount', 'date')
    # So we always use 'description' for the merge, regardless of what was passed in
    merge_key = 'description'  # Always use normalized column name
    
    # Verify the merge key exists in output_df
    if merge_key not in output_df.columns:
        # Try to find description column with different cases
        desc_candidates = [col for col in output_df.columns if col.lower() == 'description']
        if desc_candidates:
            merge_key = desc_candidates[0]
            logger.warning(f"Using '{merge_key}' instead of 'description' for merge")
        else:
            raise ValueError(
                f"Expected 'description' column not found in output DataFrame. "
                f"Available columns: {list(output_df.columns)}. "
                f"This should not happen - load_transactions should normalize column names."
            )
    
    # Verify 'original_description' exists in results_df
    if 'original_description' not in results_df.columns:
        raise ValueError(
            f"Expected 'original_description' column not found in results DataFrame. "
            f"Available columns: {list(results_df.columns)}."
        )
    
    # Final validation before merge
    if not merge_key or merge_key is None:
        raise ValueError(f"Invalid merge_key: {merge_key}. Cannot perform merge.")
    
    if merge_key not in output_df.columns:
        raise ValueError(
            f"Merge key '{merge_key}' not found in output_df. "
            f"Available columns: {list(output_df.columns)}"
        )
    
    if 'original_description' not in results_df.columns:
        raise ValueError(
            f"'original_description' not found in results_df. "
            f"Available columns: {list(results_df.columns)}"
        )
    
    # Perform the merge
    try:
        logger.debug(f"Merging: left_on='{merge_key}', right_on='original_description'")
        logger.debug(f"output_df shape: {output_df.shape}, results_df shape: {results_df[available_cols].shape}")
        output_df = output_df.merge(
            results_df[available_cols],
            left_on=merge_key,
            right_on='original_description',
            how='left'
        )
    except Exception as e:
        logger.error(f"Merge failed. output_df columns: {list(output_df.columns)}")
        logger.error(f"results_df columns: {list(results_df.columns)}")
        logger.error(f"merge_key: '{merge_key}' (type: {type(merge_key)}), available_cols: {available_cols}")
        raise ValueError(f"Failed to merge results: {e}") from e
    
    # Drop duplicate column
    if 'original_description' in output_df.columns:
        output_df = output_df.drop(columns=['original_description'])
    
    # Normalize column names to lowercase for consistency
    # This ensures the output always has consistent column names regardless of input
    column_rename_map = {}
    if 'Date' in output_df.columns and 'date' not in output_df.columns:
        column_rename_map['Date'] = 'date'
    if 'Description' in output_df.columns and 'description' not in output_df.columns:
        column_rename_map['Description'] = 'description'
    if 'Amount' in output_df.columns and 'amount' not in output_df.columns:
        column_rename_map['Amount'] = 'amount'
    
    if column_rename_map:
        output_df = output_df.rename(columns=column_rename_map)
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Processed {total} transactions in {processing_time:.2f} seconds")
    logger.info(f"✅ Average processing time: {processing_time/total*1000:.2f} ms per transaction")
    logger.info(f"✅ Results saved to {output_file}")
    
    # Print statistics
    print_statistics(output_df)
    
    # Print classification source statistics if LLM was used
    if 'classification_source' in output_df.columns:
        llm_count = (output_df['classification_source'] == 'llm').sum()
        direct_count = (output_df['classification_source'] == 'direct').sum()
        fallback_count = (output_df['classification_source'] == 'direct_fallback').sum()
        if llm_count > 0:
            print(f"\n📊 Classification Source Statistics:")
            print(f"   LLM Classification: {llm_count} ({llm_count/total*100:.1f}%)")
            print(f"   Direct Classification: {direct_count} ({direct_count/total*100:.1f}%)")
            if fallback_count > 0:
                print(f"   Direct Fallback: {fallback_count} ({fallback_count/total*100:.1f}%)")
    
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
    parser.add_argument('--use-agent', action='store_true',
                       help='[DEPRECATED] LLM is now automatically used for low-confidence matches')
    
    args = parser.parse_args()
    
    # Note: LLM is now automatically used for low-confidence matches
    # The --use-agent flag is kept for backward compatibility but does nothing
    if args.use_agent:
        logger.info("Note: LLM is now automatically enabled for low-confidence matches")
    
    print("🚀 IntelliSpend Transaction Processing Pipeline")
    print("=" * 80)
    print("🤖 Auto-LLM: ENABLED (will automatically use LLM for low-confidence matches)")
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

