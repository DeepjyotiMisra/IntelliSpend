"""
Agent Team - Agno team orchestration for IntelliSpend
This module sets up and coordinates all the specialized agents working together.
"""

from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import logging

from agents.preprocessor_agent import PreprocessorAgent
from agents.embedding_agent import EmbeddingAgent
from agents.retriever_agent import RetrieverAgent
from agents.classifier_agent import ClassifierAgent
from agents.feedback_agent import FeedbackAgent
from agents.parallel_processor import ParallelTransactionProcessor, ProcessingConfig
from models.transaction import TransactionData, CategoryPrediction
from config.config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

logger = logging.getLogger(__name__)


class IntelliSpendAgentTeam:
    """
    Agno-based agent team for IntelliSpend transaction categorization.
    
    This class orchestrates multiple specialized agents working together:
    - PreprocessorAgent: Cleans and validates transaction data
    - EmbeddingAgent: Generates vector embeddings for similarity search
    - RetrieverAgent: Performs similarity search and pattern matching
    - ClassifierAgent: Categorizes transactions using multiple strategies
    - FeedbackAgent: Handles human-in-the-loop learning and improvements
    """
    
    def __init__(self):
        self.team = None  # Will be initialized in _setup_team
        self.agents = {}
        self._initialize_agents()
        self._setup_team()
        self._initialize_vector_store_from_sample_data()
    
    def _initialize_agents(self):
        """Initialize all specialized agents."""
        try:
            logger.info("Initializing IntelliSpend agent team...")
            
            # Load environment variables (following basicAgents pattern)
            load_dotenv(Path(__file__).parent.parent / '.env')
            
            # Initialize individual agents
            self.agents = {
                'preprocessor': PreprocessorAgent(),
                'embedding': EmbeddingAgent(),
                'retriever': RetrieverAgent(),
                'classifier': ClassifierAgent(),
                'feedback': FeedbackAgent()
            }
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise e
    
    def _setup_team(self):
        """Setup the Agno team with all agents."""
        try:
            # Configure OpenAI model for Agno team
            team_model = None
            try:
                # Check for required environment variables using config
                if OPENAI_API_KEY and OPENAI_BASE_URL:
                    team_model = OpenAIChat(
                        id= MODEL_NAME,
                        base_url=OPENAI_BASE_URL,
                        extra_headers={
                            "Api-Key": OPENAI_API_KEY,
                            "Content-Type": "application/json"
                        },
                        timeout=60.0,
                        max_retries=3
                    )
                    logger.info("OpenAI model configured for Agno team")
                else:
                    logger.info("Missing OpenAI configuration - team will work without LLM coordination")
            except Exception as e:
                logger.warning(f"Failed to configure OpenAI model: {e}")
            
            # Create the Agno team with all specialized agents
            self.team = Team(
                name="IntelliSpend Agent Team",
                role="Multi-agent team for intelligent transaction categorization with human-in-the-loop learning",
                model=team_model,  # Enable LLM-powered team coordination if available
                members=[
                    self.agents['preprocessor'],
                    self.agents['embedding'], 
                    self.agents['retriever'],
                    self.agents['classifier'],
                    self.agents['feedback']
                ]
            )
            
            logger.info(f"Agent team setup complete with {len(self.agents)} agents")
            
            # Initialize parallel processor for high-performance processing
            self.parallel_processor = None
            logger.info("Parallel processing capabilities initialized")
            
        except Exception as e:
            logger.error(f"Error setting up team: {e}")
            raise e
    
    def _initialize_vector_store_from_sample_data(self):
        """Initialize vector store with merchants from sample_transactions.csv if empty."""
        try:
            from utils.vector_store import initialize_sample_merchants
            import os
            
            # Check if vector store already has data
            vector_store_path = "data/vectors/merchant_index.faiss"
            
            if not os.path.exists(vector_store_path):
                logger.info("Vector store not found. Initializing with sample merchants...")
                vector_store = self.agents['retriever'].vector_store
                initialize_sample_merchants(vector_store)
                logger.info("✅ Vector store initialized with sample merchants")
            else:
                # Check if vector store is empty
                vector_store = self.agents['retriever'].vector_store
                if vector_store.index.ntotal == 0:
                    logger.info("Vector store is empty. Initializing with sample merchants...")
                    initialize_sample_merchants(vector_store)
                    logger.info("✅ Vector store initialized with sample merchants")
                else:
                    logger.info(f"✅ Vector store already initialized with {vector_store.index.ntotal} merchants")
            
        except Exception as e:
            logger.warning(f"Could not auto-initialize vector store: {e}")
            logger.info("Vector store will be initialized on first use")
    
    def process_transactions_parallel(self, transactions: List[TransactionData], 
                                    config: ProcessingConfig = None) -> Dict[str, Any]:
        """High-performance parallel processing for transaction batches"""
        try:
            if config is None:
                # Auto-configure based on batch size
                max_workers = min(6, max(2, len(transactions) // 4))
                batch_size = min(8, max(3, len(transactions) // max_workers))
                config = ProcessingConfig(
                    max_workers=max_workers,
                    batch_size=batch_size,
                    parallel_embeddings=len(transactions) >= 8,
                    parallel_similarity=len(transactions) >= 8,
                    parallel_classification=False  # Conservative: keep sequential for accuracy
                )
            
            # Initialize parallel processor if needed
            if self.parallel_processor is None:
                self.parallel_processor = ParallelTransactionProcessor(self, config)
            
            # Process transactions in parallel
            return self.parallel_processor.process_transactions_parallel(transactions)
            
        except Exception as e:
            logger.error(f"Error in parallel processing, falling back to sequential: {e}")
            return self.process_transactions_sequential(transactions)
    


    def process_transactions(self, raw_transactions, use_parallel: bool = True) -> Dict[str, Any]:
        """Process transactions with automatic parallel processing for batches >= 5 transactions"""
        try:
            if not raw_transactions:
                return {"status": "error", "error": "No transactions provided", "results": {}}
            
            # Convert to TransactionData if needed
            if isinstance(raw_transactions[0], dict):
                transactions = [TransactionData(**tx) for tx in raw_transactions]
            else:
                transactions = raw_transactions
            
            batch_size = len(transactions)
            
            # Use parallel processing for batches >= 5 transactions
            if batch_size >= 5 and use_parallel:
                logger.info(f"🚀 Processing {batch_size} transactions in parallel mode")
                
                # Conservative parallel config
                config = ProcessingConfig(
                    max_workers=min(4, max(2, batch_size // 3)),
                    batch_size=min(6, max(3, batch_size // 2)),
                    parallel_embeddings=True,
                    parallel_similarity=True,
                    parallel_classification=False  # Keep sequential for accuracy
                )
                
                return self.process_transactions_parallel(transactions, config)
            else:
                logger.info(f"📝 Processing {batch_size} transactions sequentially")
                return self.process_transactions_sequential(transactions)
                
        except Exception as e:
            logger.error(f"Error in transaction processing: {e}")
            return {"status": "error", "error": str(e), "results": {}}

    def process_transactions_sequential(self, processed_transactions) -> Dict[str, Any]:
        """
        Original sequential transaction processing method.
        
        Args:
            processed_transactions: List of TransactionData objects
        
        Returns:
            Processing results
        """
        try:
            logger.info(f"Step 1: Processing {len(processed_transactions)} transactions sequentially")
            
            # Step 2: Generate embeddings
            logger.info("Step 2: Generating embeddings...")
            embeddings_result = self.agents['embedding'].generate_transaction_embeddings(processed_transactions)
            
            # Step 3: Find similar merchants  
            logger.info("Step 3: Finding similar merchants...")
            similarity_results = {}
            
            for transaction in processed_transactions:
                try:
                    similarities = self.agents['retriever'].search_similar_merchants(
                        transaction.merchant_name, top_k=5
                    )
                    similarity_results[transaction.id] = similarities
                except Exception as e:
                    logger.error(f"Error retrieving similarities for transaction {transaction.id}: {e}")
                    similarity_results[transaction.id] = []
            
            # Step 4: Get learned knowledge from feedback agent
            logger.info("Step 4: Retrieving learned knowledge...")
            learned_knowledge = self.agents['feedback'].export_learned_knowledge()
            known_merchants = {}
            
            # Support both merchant_mappings (enhanced) and merchant_patterns (basic)
            merchant_data = (learned_knowledge.get('merchant_mappings') or 
                           learned_knowledge.get('merchant_patterns') or {})
            
            for merchant, data in merchant_data.items():
                if isinstance(data, dict) and 'category' in data:
                    known_merchants[merchant] = data['category']
                elif isinstance(data, str):
                    # Direct category mapping
                    known_merchants[merchant] = data
            
            logger.info(f"Retrieved {len(known_merchants)} learned merchant mappings from feedback")
            
            # Step 5: Classify transactions
            logger.info("Step 5: Classifying transactions...")
            predictions = []
            
            for transaction in processed_transactions:
                similar_merchants = similarity_results.get(transaction.id, [])
                
                # Classify the transaction
                prediction = self.agents['classifier'].classify_transaction(
                    transaction, known_merchants, similar_merchants
                )
                
                # Adjust confidence based on feedback
                prediction = self.agents['feedback'].adjust_prediction_confidence(prediction, transaction)
                
                predictions.append(prediction)
            
            # Step 6: Generate human review suggestions
            logger.info("Step 6: Generating review suggestions...")
            review_suggestions = self.agents['feedback'].suggest_human_review(
                processed_transactions, predictions, confidence_threshold=0.6
            )
            
            # Step 7: Compile results
            results = {
                "processed_transactions": len(processed_transactions),
                "predictions": predictions,
                "embeddings_generated": len(embeddings_result),
                "similarity_searches": len(similarity_results),
                "learned_merchants": len(known_merchants),
                "review_suggestions": review_suggestions,
                "pipeline_stats": {
                    "preprocessing_success_rate": 1.0,  # Already preprocessed
                    "average_confidence": sum(p.confidence_score for p in predictions) / len(predictions) if predictions else 0,
                    "high_confidence_predictions": len([p for p in predictions if p.confidence_score >= 0.8]),
                    "suggestions_for_review": len(review_suggestions)
                }
            }
            
            logger.info("Transaction processing pipeline completed successfully")
            return {
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in sequential transaction processing: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "results": {}
            }

    def collect_feedback(self, transaction: TransactionData, prediction: CategoryPrediction,
                        user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Collect user feedback and learn from it."""
        try:
            # Process the feedback using the correct method
            feedback_result = self.agents['feedback'].process_feedback(
                transaction, prediction, user_feedback
            )
            
            # Update retriever's merchant mappings if feedback was successful
            if feedback_result.get('success'):
                correct_category = user_feedback.get('correct_category')
                if correct_category:
                    self.agents['retriever'].update_merchant_category(
                        transaction.merchant_name, correct_category
                    )
            
            return feedback_result
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_team_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all agents."""
        try:
            stats = {}
            
            # Get stats from each agent
            for agent_name, agent in self.agents.items():
                try:
                    if hasattr(agent, 'get_embedding_stats'):
                        stats[f"{agent_name}_stats"] = agent.get_embedding_stats()
                    elif hasattr(agent, 'get_retrieval_stats'):
                        stats[f"{agent_name}_stats"] = agent.get_retrieval_stats()
                    elif hasattr(agent, 'get_learning_metrics'):
                        stats[f"{agent_name}_stats"] = agent.get_learning_metrics()
                except Exception as e:
                    logger.error(f"Error getting stats from {agent_name}: {e}")
                    stats[f"{agent_name}_stats"] = {"error": str(e)}
            
            return {
                "team_name": "IntelliSpend Agent Team",
                "agent_count": len(self.agents),
                "agents": list(self.agents.keys()),
                "individual_stats": stats,
                "team_initialized": self.team is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting team stats: {e}")
            return {"error": str(e)}
    
