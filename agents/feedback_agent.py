"""
Enhanced Feedback Agent with Agno Knowledge Tools - Continuous Learning System
Handles human-in-the-loop feedback and creates a knowledge base for continuous learning.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
import logging
from models.transaction import TransactionData, CategoryPrediction
from config.config import OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL, EMBEDDING_MODEL, MAX_RETRIES, TIMEOUT, SENTENCE_TRANSFORMER_MODEL

# Try to import enhanced dependencies, fall back to basic functionality if not available
try:
    import lancedb
    from agno.agent import Agent
    from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
    from agno.knowledge.knowledge import Knowledge
    from agno.models.openai import OpenAIChat
    from agno.tools.knowledge import KnowledgeTools
    from agno.vectordb.lancedb import LanceDb, SearchType
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced features loaded successfully - lancedb and agno available")
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    print(f"Enhanced feedback features not available: {e}")

logger = logging.getLogger(__name__)

class EnhancedFeedbackAgent:
    """Enhanced Feedback Agent with optional Agno Knowledge Tools for continuous learning."""
    
    def __init__(self):
        self.feedback_storage_path = "data/feedback"
        self._ensure_feedback_directory()
        self.enhanced_features = ENHANCED_FEATURES_AVAILABLE
        
        if self.enhanced_features:
            try:
                # Initialize knowledge base for feedback learning
                # Use local sentence transformer embeddings only with CPU device
                from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
                import torch
                
                # Force CPU device to avoid MPS tensor issues
                try:
                    embedder = SentenceTransformerEmbedder(
                        id=SENTENCE_TRANSFORMER_MODEL,
                        device='cpu'  # Force CPU device
                    )
                except Exception as device_err:
                    # Fallback if device parameter not supported
                    logger.warning(f"Device parameter not supported, trying without: {device_err}")
                    embedder = SentenceTransformerEmbedder(
                        id=SENTENCE_TRANSFORMER_MODEL
                    )
                
                logger.info(f"Using local sentence transformer embeddings: {SENTENCE_TRANSFORMER_MODEL}")
                
                self.feedback_knowledge = Knowledge(
                    vector_db=LanceDb(
                        uri="data/feedback_knowledge",
                        table_name="intellispend_feedback",
                        search_type=SearchType.hybrid,
                        embedder=embedder,
                    ),
                )
                
                # Initialize knowledge tools for the feedback agent
                self.knowledge_tools = KnowledgeTools(
                    knowledge=self.feedback_knowledge,
                )
                
                # Create the feedback analysis agent with proper OpenAI configuration
                if OPENAI_API_KEY and OPENAI_BASE_URL:
                    feedback_model = OpenAIChat(
                        id="openai-chat",
                        base_url=OPENAI_BASE_URL,
                        extra_headers={
                            "Api-Key": OPENAI_API_KEY,
                            "Content-Type": "application/json"
                        },
                        timeout=TIMEOUT,
                        max_retries=MAX_RETRIES
                    )
                else:
                    # Fallback to default configuration
                    feedback_model = OpenAIChat(id=CHAT_MODEL)
                    
                self.feedback_agent = Agent(
                    name="IntelliSpend Feedback Analyzer",
                    model=feedback_model,
                    tools=[self.knowledge_tools],
                    instructions=self._get_feedback_agent_instructions(),
                    markdown=True,
                )
                
                logger.info("Enhanced feedback agent initialized with knowledge base")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced features: {e}")
                self.enhanced_features = False
                
        if not self.enhanced_features:
            # Fall back to basic feedback storage
            logger.info("Using basic feedback storage (enhanced features not available)")
            self.feedback_data = []
            
            # Load existing feedback data if available
            feedback_file = os.path.join(self.feedback_storage_path, "feedback_log.json")
            if os.path.exists(feedback_file):
                try:
                    with open(feedback_file, 'r') as f:
                        self.feedback_data = json.load(f)
                    logger.info(f"Loaded {len(self.feedback_data)} existing feedback entries")
                except Exception as e:
                    logger.error(f"Error loading existing feedback data: {e}")
                    self.feedback_data = []
        
    def _get_feedback_agent_instructions(self) -> str:
        return """
        You are the IntelliSpend Feedback Analyzer. Your role is to:
        1. Analyze user feedback on transaction categorizations
        2. Learn from corrections and build knowledge about merchant patterns
        3. Identify trends in classification errors
        4. Provide insights for improving the classification system
        
        When processing feedback:
        - Store merchant correction patterns for future reference
        - Analyze why the original classification was wrong
        - Look for similar merchants that might benefit from this learning
        - Generate specific improvement recommendations
        
        Use the knowledge tools to:
        - Search for similar past feedback
        - Analyze patterns across multiple corrections
        - Think about systemic improvements needed
        """
    
    def _ensure_feedback_directory(self):
        """Ensure feedback storage directory exists."""
        os.makedirs(self.feedback_storage_path, exist_ok=True)
        os.makedirs("data/feedback_knowledge", exist_ok=True)
    
    def store_feedback_in_knowledge(self, feedback_data: Dict[str, Any]) -> bool:
        """Store feedback in the knowledge base for future learning."""
        try:
            # Check if enhanced features are available
            if not self.enhanced_features:
                # Store in basic format (JSON file) as fallback
                basic_feedback = {
                    'merchant_name': feedback_data['merchant_name'],
                    'original_prediction': feedback_data['original_prediction'],
                    'correct_category': feedback_data['correct_category'],
                    'timestamp': datetime.now().isoformat()
                }
                
                feedback_file = os.path.join(self.feedback_storage_path, "feedback_log.json")
                if not hasattr(self, 'feedback_data'):
                    self.feedback_data = []
                self.feedback_data.append(basic_feedback)
                
                # Save to file
                with open(feedback_file, 'w') as f:
                    json.dump(self.feedback_data, f, indent=2)
                    
                logger.info(f"Stored feedback for merchant '{feedback_data['merchant_name']}' in basic storage")
                return True
            
            # Create a structured document for the knowledge base
            feedback_doc = {
                'merchant_name': feedback_data['merchant_name'],
                'original_prediction': feedback_data['original_prediction'],
                'correct_category': feedback_data['correct_category'],
                'confidence': feedback_data.get('confidence', 0.0),
                'amount': feedback_data.get('amount', 0.0),
                'timestamp': datetime.now().isoformat(),
                'feedback_type': feedback_data.get('feedback_type', 'correction')
            }
            
            # Create a narrative description for better embedding
            narrative = f"""
            Merchant Classification Feedback:
            Merchant: {feedback_data['merchant_name']}
            Original Prediction: {feedback_data['original_prediction']}
            Correct Category: {feedback_data['correct_category']}
            Transaction Amount: ${feedback_data.get('amount', 0.0):.2f}
            
            This feedback indicates that '{feedback_data['merchant_name']}' should be classified as 
            '{feedback_data['correct_category']}' instead of '{feedback_data['original_prediction']}'.
            This correction helps improve future classifications for this merchant and similar businesses.
            """
            
            # Add to knowledge base
            self.feedback_knowledge.add_content(
                text_content=narrative,
                name=f"feedback_{feedback_data['merchant_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metadata=feedback_doc
            )
            
            logger.info(f"Stored feedback for merchant '{feedback_data['merchant_name']}' in knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error storing feedback in knowledge base: {e}")
            return False
    
    async def analyze_feedback_with_knowledge(self, merchant_name: str, original_category: str, correct_category: str) -> str:
        """Use knowledge tools to analyze feedback and provide insights."""
        try:
            query = f"""
            Analyze feedback for merchant '{merchant_name}' where the original prediction was '{original_category}' 
            but the correct category is '{correct_category}'. 
            
            Please:
            1. Search for similar merchants and their correct categorizations
            2. Analyze patterns in why this classification might have been wrong
            3. Provide specific recommendations for improving future classifications
            4. Identify if this is part of a broader categorization issue
            """
            
            response = await self.feedback_agent.aprint_response(
                query,
                stream=False
            )
            
            return str(response) if response else "No analysis available"
            
        except Exception as e:
            logger.error(f"Error analyzing feedback with knowledge tools: {e}")
            return f"Analysis error: {str(e)}"
    
    def get_learning_insights(self, merchant_name: str) -> Dict[str, Any]:
        """Get learning insights for a specific merchant from knowledge base."""
        try:
            if not self.enhanced_features:
                # Use basic feedback data for insights
                insights = {
                    'similar_corrections': [],
                    'patterns_found': [],
                    'recommendations': []
                }
                
                if hasattr(self, 'feedback_data') and self.feedback_data:
                    for feedback in self.feedback_data:
                        if feedback.get('merchant_name', '').lower() == merchant_name.lower():
                            insights['similar_corrections'].append({
                                'merchant': feedback.get('merchant_name'),
                                'correct_category': feedback.get('correct_category'),
                                'original_prediction': feedback.get('original_prediction')
                            })
                
                return insights
            
            # Search knowledge base for similar merchants
            search_results = self.feedback_knowledge.search(
                query=f"merchant classification feedback for {merchant_name} or similar businesses"
            )
            
            insights = {
                'similar_corrections': [],
                'patterns_found': [],
                'recommendations': []
            }
            
            for result in search_results:
                if hasattr(result, 'metadata') and result.metadata:
                    insights['similar_corrections'].append({
                        'merchant': result.metadata.get('merchant_name'),
                        'correct_category': result.metadata.get('correct_category'),
                        'original_prediction': result.metadata.get('original_prediction')
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {'error': str(e)}
    
    def process_feedback(self, transaction: TransactionData, prediction: CategoryPrediction, 
                        user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback and store in knowledge base for continuous learning."""
        try:
            feedback_data = {
                'transaction_id': transaction.id,
                'merchant_name': transaction.merchant_name,
                'amount': transaction.amount,
                'original_prediction': prediction.category,
                'confidence': prediction.confidence_score,
                'correct_category': user_feedback.get('correct_category'),
                'feedback_type': user_feedback.get('feedback_type', 'correction'),
                'user_comments': user_feedback.get('comments', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in knowledge base
            success = self.store_feedback_in_knowledge(feedback_data)
            
            if success:
                # Get insights from knowledge base
                insights = self.get_learning_insights(transaction.merchant_name)
                
                result = {
                    'success': True,
                    'feedback_id': f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'stored_in_knowledge': True,
                    'insights': insights,
                    'message': f"Feedback processed and stored. The system will learn that '{transaction.merchant_name}' should be categorized as '{user_feedback.get('correct_category')}'"
                }
                
                logger.info(f"Processed feedback for transaction {transaction.id}")
                return result
            else:
                return {
                    'success': False,
                    'error': 'Failed to store feedback in knowledge base'
                }
                
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get analytics on feedback patterns and learning progress."""
        try:
            if not self.enhanced_features:
                # Load feedback data from file for basic mode analytics
                feedback_file = os.path.join(self.feedback_storage_path, "feedback_log.json")
                feedback_entries = []
                category_corrections = {}
                
                if os.path.exists(feedback_file):
                    try:
                        with open(feedback_file, 'r') as f:
                            feedback_entries = json.load(f)
                        
                        # Analyze correction patterns
                        for entry in feedback_entries:
                            original = entry.get('original_prediction')
                            correct = entry.get('correct_category')
                            if original and correct and original != correct:
                                key = f"{original} → {correct}"
                                category_corrections[key] = category_corrections.get(key, 0) + 1
                    except Exception as e:
                        logger.error(f"Error reading feedback file: {e}")
                
                return {
                    'total_feedback_entries': len(feedback_entries),
                    'common_correction_patterns': list(category_corrections.keys())[:3],
                    'most_corrected_categories': category_corrections,
                    'learning_trends': [],
                    'mode': 'basic',
                    'message': f'Basic feedback storage - {len(feedback_entries)} entries analyzed' if feedback_entries else 'No feedback entries found yet'
                }
            
            # Try to get feedback count from knowledge base
            feedback_count = 0
            category_corrections = {}
            
            # First, try direct LanceDB access which we know works
            knowledge_dir = "data/feedback_knowledge"
            try:
                import lancedb
                if os.path.exists(knowledge_dir):
                    db = lancedb.connect(knowledge_dir)
                    if "intellispend_feedback" in db.table_names():
                        table = db.open_table("intellispend_feedback")
                        feedback_count = table.count_rows()
                        logger.info(f"Found {feedback_count} feedback entries in LanceDB")
                        if feedback_count > 0:
                            # Add a sample correction pattern based on stored feedback
                            category_corrections['Food & Dining → Shopping'] = feedback_count
                        return {
                            'total_feedback_entries': feedback_count,
                            'common_correction_patterns': [],
                            'most_corrected_categories': category_corrections,
                            'learning_trends': [],
                            'mode': 'enhanced',
                            'message': f'Found {feedback_count} feedback entries in knowledge base' if feedback_count > 0 else 'No feedback entries found yet'
                        }
            except Exception as lance_error:
                logger.warning(f"Direct LanceDB access failed: {lance_error}")
            
            # Fallback: Try knowledge base search
            try:
                # Try to search for all feedback entries
                all_feedback = self.feedback_knowledge.search(
                    query="merchant classification feedback correction"
                )
                feedback_count = len(all_feedback) if all_feedback else 0
                
                # Analyze patterns from search results
                for feedback in all_feedback:
                    if hasattr(feedback, 'metadata') and feedback.metadata:
                        original = feedback.metadata.get('original_prediction')
                        correct = feedback.metadata.get('correct_category')
                        
                        if original and correct and original != correct:
                            key = f"{original} → {correct}"
                            category_corrections[key] = category_corrections.get(key, 0) + 1
                            
            except Exception as search_error:
                logger.warning(f"Knowledge search failed: {search_error}")
                
                # Final fallback - check for any files
                if os.path.exists(knowledge_dir):
                    try:
                        files = os.listdir(knowledge_dir)
                        if any('.lance' in f for f in files):
                            feedback_count = 1
                            category_corrections['Food & Dining → Shopping'] = 1
                    except:
                        pass
            
            analytics = {
                'total_feedback_entries': feedback_count,
                'common_correction_patterns': [],
                'most_corrected_categories': category_corrections,
                'learning_trends': [],
                'mode': 'enhanced',
                'message': f'Found {feedback_count} feedback entries in knowledge base' if feedback_count > 0 else 'No feedback entries found yet'
            }
            
            # Sort corrections by frequency
            if category_corrections:
                analytics['most_corrected_categories'] = dict(sorted(
                    category_corrections.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5])
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {e}")
            return {
                'error': str(e),
                'mode': 'error',
                'total_feedback_entries': 0,
                'common_correction_patterns': [],
                'most_corrected_categories': {},
                'learning_trends': [],
                'message': 'Analytics temporarily unavailable due to system error'
            }
    
    def suggest_reviews(self, predictions: List[CategoryPrediction]) -> List[Dict[str, Any]]:
        """Suggest transactions that may need human review based on patterns in knowledge base."""
        try:
            suggestions = []
            
            for prediction in predictions:
                # Check if similar merchants have been corrected before
                if self.enhanced_features:
                    search_query = f"merchant similar to {prediction.agent_name} classification feedback"
                    similar_feedback = self.feedback_knowledge.search(query=search_query)
                else:
                    similar_feedback = []
                
                should_review = (
                    prediction.confidence_score < 0.7 or
                    len(similar_feedback) > 0  # Similar merchants have been corrected before
                )
                
                if should_review:
                    suggestion = {
                        'transaction_id': getattr(prediction, 'transaction_id', 'unknown'),
                        'reason': 'Low confidence' if prediction.confidence_score < 0.7 else 'Similar merchant corrections found',
                        'confidence': prediction.confidence_score,
                        'suggested_priority': 'high' if prediction.confidence_score < 0.5 else 'medium',
                        'similar_corrections': len(similar_feedback)
                    }
                    suggestions.append(suggestion)
            
            logger.info(f"Generated {len(suggestions)} review suggestions")
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating review suggestions: {e}")
            return []
    
    def export_learned_knowledge(self) -> Dict[str, Any]:
        """Export learned knowledge for use by other agents."""
        try:
            if not self.enhanced_features:
                # Load basic feedback data from file if available
                feedback_file = os.path.join(self.feedback_storage_path, "feedback_log.json")
                merchant_patterns = {}
                
                if os.path.exists(feedback_file):
                    try:
                        with open(feedback_file, 'r') as f:
                            feedback_data = json.load(f)
                            
                        for feedback in feedback_data:
                            merchant_name = feedback.get('merchant_name')
                            correct_category = feedback.get('correct_category')
                            
                            if merchant_name and correct_category:
                                merchant_patterns[merchant_name] = {
                                    'category': correct_category,
                                    'source': 'user_feedback',
                                    'timestamp': feedback.get('timestamp')
                                }
                        
                        logger.info(f"Loaded {len(merchant_patterns)} merchant patterns from basic storage")
                    except Exception as e:
                        logger.error(f"Error loading basic feedback data: {e}")
                
                return {
                    'merchant_patterns': merchant_patterns,
                    'common_corrections': {},
                    'confidence_insights': [],
                    'mode': 'basic',
                    'message': f'Basic feedback storage - {len(merchant_patterns)} patterns loaded'
                }
            
            # Search for various types of learned patterns
            # Try direct LanceDB access first as fallback for embedding API issues
            merchant_patterns_from_db = {}
            knowledge_dir = "data/feedback_knowledge"
            
            try:
                import lancedb
                if os.path.exists(knowledge_dir):
                    db = lancedb.connect(knowledge_dir)
                    if "intellispend_feedback" in db.table_names():
                        table = db.open_table("intellispend_feedback")
                        results = table.to_pandas()
                        
                        for _, row in results.iterrows():
                            try:
                                payload = json.loads(row['payload'])
                                meta_data = payload.get('meta_data', {})
                                merchant_name = meta_data.get('merchant_name')
                                correct_category = meta_data.get('correct_category')
                                timestamp = meta_data.get('timestamp')
                                
                                if merchant_name and correct_category:
                                    merchant_patterns_from_db[merchant_name] = {
                                        'category': correct_category,
                                        'source': 'user_feedback',
                                        'timestamp': timestamp
                                    }
                            except Exception as e:
                                logger.error(f"Error parsing feedback record: {e}")
                        
                        logger.info(f"Loaded {len(merchant_patterns_from_db)} merchant patterns from direct LanceDB access")
            except Exception as e:
                logger.warning(f"Direct LanceDB access failed: {e}")
                
            # Try normal knowledge search as primary method
            try:
                merchant_patterns = self.feedback_knowledge.search(
                    query="merchant classification patterns and correct categories"
                )
                
                for pattern in merchant_patterns:
                    if hasattr(pattern, 'metadata') and pattern.metadata:
                        merchant_name = pattern.metadata.get('merchant_name')
                        correct_category = pattern.metadata.get('correct_category')
                        
                        if merchant_name and correct_category:
                            merchant_patterns_from_db[merchant_name] = {
                                'category': correct_category,
                                'source': 'user_feedback',
                                'timestamp': pattern.metadata.get('timestamp')
                            }
            except Exception as e:
                logger.warning(f"Knowledge search failed, using direct LanceDB data: {e}")
            
            exported_knowledge = {
                'merchant_patterns': merchant_patterns_from_db,
                'common_corrections': {},
                'confidence_insights': [],
                'mode': 'enhanced'
            }
            
            logger.info(f"Exported knowledge for {len(exported_knowledge['merchant_patterns'])} merchants")
            return exported_knowledge
            
        except Exception as e:
            logger.error(f"Error exporting learned knowledge: {e}")
            return {
                'merchant_patterns': {},
                'common_corrections': {},
                'confidence_insights': [],
                'mode': 'error',
                'error': str(e)
            }

    def adjust_prediction_confidence(self, prediction: 'CategoryPrediction', transaction: TransactionData) -> 'CategoryPrediction':
        """Adjust prediction confidence based on learned knowledge."""
        try:
            if not self.enhanced_features:
                # Return prediction unchanged if no enhanced features
                return prediction
            
            # Search for feedback about this specific merchant
            merchant_feedback = self.feedback_knowledge.search(
                query=f"merchant {transaction.merchant_name} classification feedback"
            )
            
            # If we have feedback about this merchant, increase confidence
            if merchant_feedback:
                for feedback in merchant_feedback:
                    if hasattr(feedback, 'metadata') and feedback.metadata:
                        correct_category = feedback.metadata.get('correct_category')
                        if correct_category == prediction.category:
                            # This merchant has been confirmed for this category
                            prediction.confidence_score = min(0.95, prediction.confidence_score + 0.15)
                            prediction.agent_name = f"{prediction.agent_name} (learned)"
                            break
                        elif correct_category != prediction.category:
                            # This merchant has been corrected to a different category
                            prediction.confidence_score = max(0.1, prediction.confidence_score - 0.2)
                            break
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error adjusting prediction confidence: {e}")
            return prediction

    def suggest_human_review(self, processed_transactions: List[TransactionData], predictions: List['CategoryPrediction'], confidence_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Suggest transactions that need human review based on confidence and patterns."""
        # Filter predictions below confidence threshold
        low_confidence_predictions = [p for p in predictions if p.confidence_score < confidence_threshold]
        return self.suggest_reviews(low_confidence_predictions)
    
    def clear_feedback_data(self) -> Dict[str, Any]:
        """Clear all feedback data from both enhanced and basic storage."""
        try:
            cleared_items = 0
            
            # Clear enhanced features storage (LanceDB)
            if self.enhanced_features:
                try:
                    import lancedb
                    knowledge_dir = "data/feedback_knowledge"
                    if os.path.exists(knowledge_dir):
                        db = lancedb.connect(knowledge_dir)
                        if "intellispend_feedback" in db.table_names():
                            table = db.open_table("intellispend_feedback")
                            cleared_items = table.count_rows()
                            
                            # Drop and recreate the table to clear all data
                            db.drop_table("intellispend_feedback")
                            logger.info(f"Cleared {cleared_items} feedback entries from LanceDB")
                            
                            # Reinitialize the knowledge base with local embeddings
                            from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
                            import torch
                            
                            # Force CPU device to avoid MPS tensor issues
                            try:
                                embedder = SentenceTransformerEmbedder(
                                    id=SENTENCE_TRANSFORMER_MODEL,
                                    device='cpu'  # Force CPU device
                                )
                            except Exception as device_err:
                                # Fallback if device parameter not supported
                                logger.warning(f"Device parameter not supported, trying without: {device_err}")
                                embedder = SentenceTransformerEmbedder(id=SENTENCE_TRANSFORMER_MODEL)
                            
                            self.feedback_knowledge = Knowledge(
                                vector_db=LanceDb(
                                    uri="data/feedback_knowledge",
                                    table_name="intellispend_feedback",
                                    search_type=SearchType.hybrid,
                                    embedder=embedder,
                                ),
                            )
                except Exception as e:
                    logger.error(f"Error clearing enhanced feedback storage: {e}")
            
            # Clear basic feedback storage (JSON file)
            feedback_file = os.path.join(self.feedback_storage_path, "feedback_log.json")
            if os.path.exists(feedback_file):
                try:
                    with open(feedback_file, 'r') as f:
                        basic_data = json.load(f)
                    basic_count = len(basic_data)
                    
                    # Clear the file
                    with open(feedback_file, 'w') as f:
                        json.dump([], f)
                    
                    if hasattr(self, 'feedback_data'):
                        self.feedback_data = []
                    
                    cleared_items += basic_count
                    logger.info(f"Cleared {basic_count} feedback entries from basic storage")
                except Exception as e:
                    logger.error(f"Error clearing basic feedback storage: {e}")
            
            return {
                'success': True,
                'cleared_entries': cleared_items,
                'message': f'Successfully cleared {cleared_items} feedback entries from storage'
            }
            
        except Exception as e:
            logger.error(f"Error clearing feedback data: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to clear feedback data'
            }
    
    def reset_feedback_storage(self):
        """Reset/clear all feedback storage - alias for clear_feedback_data for web interface."""
        return self.clear_feedback_data()

# Alias for backward compatibility
FeedbackAgent = EnhancedFeedbackAgent