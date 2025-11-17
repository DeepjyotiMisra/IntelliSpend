"""
Parallel Transaction Processor - High-performance parallel processing for IntelliSpend
"""

import concurrent.futures
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
from models.transaction import TransactionData, CategoryPrediction
import time

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for parallel processing"""
    max_workers: int = 4
    batch_size: int = 8
    parallel_embeddings: bool = True
    parallel_similarity: bool = True
    parallel_classification: bool = False  # Conservative: keep sequential for accuracy

class ParallelTransactionProcessor:
    """High-performance parallel processor for transactions"""
    
    def __init__(self, agent_team, config: ProcessingConfig = None):
        self.agent_team = agent_team
        self.config = config or ProcessingConfig()
        logger.info(f"Initialized parallel processor: {self.config.max_workers} workers, batch_size={self.config.batch_size}")
    
    def process_transactions_parallel(self, transactions: List[TransactionData]) -> Dict[str, Any]:
        """Main parallel processing method"""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Processing {len(transactions)} transactions in parallel (workers={self.config.max_workers})")
            
            # Preprocess transactions
            processed_transactions = self.agent_team.agents['preprocessor'].preprocess_batch(transactions)
            
            # Generate embeddings (parallel or sequential based on config)
            embeddings_result = self._generate_embeddings_parallel(processed_transactions)
            
            # Similarity search (parallel or sequential based on config)
            similarity_results = self._perform_similarity_search_parallel(processed_transactions)
            
            # Get learned knowledge
            learned_knowledge = self.agent_team.agents['feedback'].export_learned_knowledge()
            known_merchants = self._extract_merchant_mappings(learned_knowledge)
            
            # Classify transactions (parallel or sequential based on config)
            predictions = self._classify_transactions_parallel(
                processed_transactions, known_merchants, similarity_results
            )
            
            # Generate review suggestions
            review_suggestions = self.agent_team.agents['feedback'].suggest_human_review(
                processed_transactions, predictions, confidence_threshold=0.6
            )
            
            # Calculate results
            processing_time = time.time() - start_time
            transactions_per_second = len(transactions) / processing_time
            
            logger.info(f"⚡ Completed: {transactions_per_second:.1f} trans/sec in {processing_time:.2f}s")
            
            results = {
                "processed_transactions": len(processed_transactions),
                "predictions": predictions,
                "review_suggestions": review_suggestions,
                "performance_metrics": {
                    "processing_time_seconds": processing_time,
                    "transactions_per_second": transactions_per_second,
                    "parallel_workers": self.config.max_workers
                },
                "pipeline_stats": {
                    "preprocessing_success_rate": 1.0,
                    "average_confidence": sum(p.confidence_score for p in predictions) / len(predictions) if predictions else 0,
                    "high_confidence_predictions": len([p for p in predictions if p.confidence_score >= 0.8]),
                    "suggestions_for_review": len(review_suggestions)
                }
            }
            
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            return {"status": "error", "error": str(e), "results": {}}
    
    def _preprocess_transactions(self, transactions: List[TransactionData]) -> List[TransactionData]:
        """Preprocess transactions - sequential is fast enough"""
        return self.agent_team.agents['preprocessor'].preprocess_batch(transactions)
    
    def _generate_embeddings_parallel(self, transactions: List[TransactionData]) -> List[Dict[str, Any]]:
        """Generate embeddings in parallel or sequential based on config"""
        if not self.config.parallel_embeddings:
            return self.agent_team.agents['embedding'].generate_transaction_embeddings(transactions)
        
        # Split into batches for parallel processing
        batches = self._create_batches(transactions, self.config.batch_size)
        
        def process_batch(batch):
            return self.agent_team.agents['embedding'].generate_transaction_embeddings(batch)
        
        # Process batches in parallel
        all_embeddings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures, timeout=30):
                all_embeddings.extend(future.result())
        
        return all_embeddings
    
    def _perform_similarity_search_parallel(self, transactions: List[TransactionData]) -> Dict[str, List]:
        """Perform similarity search in parallel or sequential based on config"""
        if not self.config.parallel_similarity:
            similarity_results = {}
            for transaction in transactions:
                try:
                    similarities = self.agent_team.agents['retriever'].vector_store.find_similar_merchants(
                        transaction.merchant_name, k=5
                    )
                    similarity_results[transaction.id] = similarities
                except Exception as e:
                    logger.error(f"Error in similarity search for {transaction.id}: {e}")
                    similarity_results[transaction.id] = []
            return similarity_results
        
        # Parallel similarity search
        def search_transaction(transaction):
            try:
                similarities = self.agent_team.agents['retriever'].vector_store.find_similar_merchants(
                    transaction.merchant_name, k=5
                )
                return transaction.id, similarities
            except Exception as e:
                logger.error(f"Error in similarity search for {transaction.id}: {e}")
                return transaction.id, []
        
        similarity_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(search_transaction, t) for t in transactions]
            for future in concurrent.futures.as_completed(futures, timeout=30):
                transaction_id, similarities = future.result()
                similarity_results[transaction_id] = similarities
        
        return similarity_results
    
    def _classify_transactions_parallel(self, transactions: List[TransactionData], 
                                      known_merchants: Dict[str, str], 
                                      similarity_results: Dict[str, List]) -> List[CategoryPrediction]:
        """Classify transactions in parallel or sequential based on config"""
        if not self.config.parallel_classification:
            # Sequential classification (recommended for accuracy)
            predictions = []
            for transaction in transactions:
                similar_merchants = similarity_results.get(transaction.id, [])
                prediction = self.agent_team.agents['classifier'].classify_transaction(
                    transaction, known_merchants, similar_merchants
                )
                prediction = self.agent_team.agents['feedback'].adjust_prediction_confidence(prediction, transaction)
                predictions.append(prediction)
            return predictions
        
        # Parallel classification (experimental)
        def classify_batch(batch_transactions):
            batch_predictions = []
            for transaction in batch_transactions:
                try:
                    similar_merchants = similarity_results.get(transaction.id, [])
                    prediction = self.agent_team.agents['classifier'].classify_transaction(
                        transaction, known_merchants, similar_merchants
                    )
                    prediction = self.agent_team.agents['feedback'].adjust_prediction_confidence(prediction, transaction)
                    batch_predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Error classifying transaction {transaction.id}: {e}")
                    fallback_prediction = CategoryPrediction(
                        category="Other", confidence_score=0.1, agent_name="error_fallback",
                        reasoning=f"Classification error: {str(e)}"
                    )
                    batch_predictions.append(fallback_prediction)
            return batch_predictions
        
        batches = self._create_batches(transactions, self.config.batch_size)
        all_predictions = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(classify_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures, timeout=30):
                all_predictions.extend(future.result())
        
        return all_predictions
    
    def _extract_merchant_mappings(self, learned_knowledge: Dict[str, Any]) -> Dict[str, str]:
        """Extract merchant mappings from learned knowledge"""
        known_merchants = {}
        merchant_data = (learned_knowledge.get('merchant_mappings') or 
                        learned_knowledge.get('merchant_patterns') or {})
        
        for merchant, data in merchant_data.items():
            if isinstance(data, dict) and 'category' in data:
                known_merchants[merchant] = data['category']
            elif isinstance(data, str):
                known_merchants[merchant] = data
        
        return known_merchants
    
    def _create_batches(self, items: List, batch_size: int) -> List[List]:
        """Split items into batches for parallel processing"""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

