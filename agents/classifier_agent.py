"""
Classifier Agent - Performs transaction categorization in IntelliSpend
This agent implements multiple classification strategies including exact matching,
similarity search, rule-based classification, and LLM-based reasoning.
"""

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools import tool
from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from models.transaction import TransactionData, CategoryPrediction
from config.config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

logger = logging.getLogger(__name__)


class ClassifierAgent(Agent):
    """
    Agent responsible for transaction classification using multiple strategies.
    
    Key responsibilities:
    - Exact merchant matching
    - Rule-based classification
    - Similarity-based classification
    - LLM-powered intelligent classification
    - Confidence scoring
    """
    
    def __init__(self):
        super().__init__(
            name="ClassifierAgent",
            description="Classifies transactions using multiple intelligent strategies with advanced analysis tools",
            instructions=[
                "Apply exact merchant matching for known merchants",
                "Use rule-based classification for pattern recognition", 
                "Leverage similarity search for merchant matching",
                "Apply LLM reasoning for complex cases",
                "Use custom tools for merchant analysis and confidence validation",
                "Provide confidence scores for all predictions"
            ],
            tools=[
                DuckDuckGoTools()  # Web search for unknown merchants
            ]
        )
        
        # Standard spending categories
        self.standard_categories = [
            "Food & Dining", "Shopping", "Transportation", "Bills & Utilities",
            "Entertainment", "Health & Medical", "Travel", "Education",
            "Personal Care", "Financial", "Business", "Gifts & Donations",
            "Home & Garden", "Auto & Transport", "Insurance", "Taxes",
            "Investments", "Other"
        ]
        
        # Available categories
        self.categories = [
            "Food & Dining", "Transportation", "Shopping", "Entertainment",
            "Bills & Utilities", "Financial", "Health & Medical", 
            "Home & Garden", "Travel", "Education", "Income", "Other"
        ]
        
        # Classification rules
        self.merchant_rules = self._load_merchant_rules()
        self.amount_rules = self._load_amount_rules()
        self.description_rules = self._load_description_rules()
        
        # Agno OpenAI model for LLM classification
        self.model = None
        self._initialize_llm_model()
    
    def _initialize_llm_model(self):
        """Initialize OpenAI client for OpenAI models."""
        try:
            # Use configuration from config file
            if OPENAI_API_KEY and OPENAI_BASE_URL:
                # Initialize using basicAgents working pattern with Agno OpenAIChat
                from agno.models.openai import OpenAIChat
                
                # Create Agno OpenAI model with basicAgents pattern
                self.model = OpenAIChat(
                    id= MODEL_NAME,
                    base_url=OPENAI_BASE_URL,
                    extra_headers={
                        "Api-Key": OPENAI_API_KEY,
                        "Content-Type": "application/json"
                    },
                    timeout=60.0,
                    max_retries=3
                )
                
                logger.info("OpenAI model initialized for classification agent")
            else:
                logger.warning("Missing OpenAI configuration - classifier will use rule-based methods only")
                self.model = None
            
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI model: {e}")
            self.model = None
    
    def _load_merchant_rules(self) -> Dict[str, str]:
        """Load merchant-to-category mapping rules."""
        return {
            # Food & Dining
            r'(?i)(mcdonald\'?s|mcdonalds|burger king|kfc|taco bell|subway|pizza|restaurant|cafe|starbucks|dunkin)': 'Food & Dining',
            r'(?i)(grocery|supermarket|walmart|target|safeway|kroger|whole foods)': 'Food & Dining',
            
            # Transportation
            r'(?i)(uber|lyft|taxi|gas|shell|exxon|bp|chevron|mobil|parking)': 'Transportation',
            r'(?i)(airline|airport|flight|car rental|train|bus)': 'Transportation',
            
            # Shopping
            r'(?i)(amazon|ebay|shop|store|retail|mall|clothing|fashion)': 'Shopping',
            
            # Entertainment
            r'(?i)(netflix|spotify|hulu|disney|movie|theater|game|music)': 'Entertainment',
            
            # Bills & Utilities
            r'(?i)(electric|gas company|water|internet|phone|cable|utility)': 'Bills & Utilities',
            
            # Financial
            r'(?i)(bank|atm|fee|interest|payment|transfer|loan)': 'Financial',
            
            # Health & Medical
            r'(?i)(hospital|doctor|medical|pharmacy|health|dental)': 'Health & Medical',
            
            # Home & Garden
            r'(?i)(home depot|lowes|hardware|furniture|garden|repair)': 'Home & Garden'
        }
    
    def _load_amount_rules(self) -> List[Tuple[float, float, str, float]]:
        """Load amount-based classification rules."""
        return [
            # (min_amount, max_amount, category, confidence)
            (0.99, 15.99, 'Food & Dining', 0.6),  # Small food purchases
            (200, 1000, 'Shopping', 0.5),         # Medium shopping
            (1000, float('inf'), 'Major Purchase', 0.4),  # Large purchases
            (0.01, 5.00, 'Fees', 0.7),           # Small fees
        ]
    
    def _load_description_rules(self) -> Dict[str, str]:
        """Load description pattern rules."""
        return {
            r'(?i)(coffee|lunch|dinner|breakfast)': 'Food & Dining',
            r'(?i)(gas|fuel|gasoline)': 'Transportation',
            r'(?i)(subscription|monthly|recurring)': 'Bills & Utilities',
            r'(?i)(withdrawal|deposit|transfer)': 'Financial',
            r'(?i)(fee|charge|penalty)': 'Financial',
            r'(?i)(refund|return|credit)': 'Other'
        }
    

    

    
    def similarity_based_classification(self, transaction: TransactionData, 
                                      similar_merchants: List[Tuple[str, float, str]]) -> Optional[CategoryPrediction]:
        """Classify based on similar merchant matches."""
        try:
            if not similar_merchants:
                return None
            
            # Use the most similar merchant
            best_merchant, similarity, category = similar_merchants[0]
            
            # Only use if similarity is high enough
            if similarity >= 0.7:
                confidence = min(0.85, similarity * 0.9)  # Scale down confidence slightly
                
                return CategoryPrediction(
                    category=category,
                    confidence_score=confidence,
                    agent_name="similarity_match",
                    reasoning=f"Similar to '{best_merchant}' (similarity: {similarity:.3f})"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in similarity-based classification: {e}")
            return None
    
    def web_search_classification(self, transaction: TransactionData) -> Optional[CategoryPrediction]:
        """Use web search to classify unknown merchants when confidence is low - leverages LLM tools."""
        try:
            # Clean merchant name for search
            clean_merchant = transaction.merchant_name.strip().replace('#', '').replace('*', '')
            
            logger.info(f"Triggering LLM-based web search for merchant: {clean_merchant}")
            
            # Use LLM with tools to perform web search and classification
            if self.model:
                try:
                    from agno.models.message import Message
                    
                    # Create a prompt that encourages the LLM to use web search tools
                    search_prompt = f"""
                    I need to classify the merchant "{transaction.merchant_name}" for a transaction of ${abs(transaction.amount):.2f}.
                    
                    The merchant is unknown and I need to determine which category it belongs to from: {', '.join(self.categories)}
                    
                    Please use web search to find information about this merchant's business type and classify it appropriately.
                    Consider the transaction amount as additional context.
                    
                    Respond with just the category name followed by your confidence (1-10).
                    Example: "Shopping 8"
                    """
                    
                    messages = [
                        Message(role="system", content="You are a transaction categorization expert with access to web search tools. Use DuckDuckGo search to research unknown merchants and classify them accurately."),
                        Message(role="user", content=search_prompt)
                    ]
                    
                    # Let the LLM use tools to research the merchant
                    response = self.model.invoke(messages, assistant_message=Message(role="assistant", content=""))
                    response_text = response.content.strip()
                    
                    # Parse the response to extract category and reasoning
                    category, confidence, reasoning = self._parse_web_search_response(response_text, clean_merchant)
                    
                    logger.info(f"LLM-based web search result for {clean_merchant}: {category} ({confidence:.2f})")
                    
                    return CategoryPrediction(
                        category=category,
                        confidence_score=confidence,
                        agent_name="llm_with_web_search",
                        reasoning=f"LLM web search classification: {reasoning}"
                    )
                    
                except Exception as llm_error:
                    logger.warning(f"LLM-based web search failed: {llm_error}")
            
            # Fallback to simple pattern analysis if LLM web search fails
            merchant_lower = clean_merchant.lower()
            
            # Simple pattern-based category inference
            if any(term in merchant_lower for term in ['cafe', 'coffee', 'restaurant', 'kitchen', 'deli', 'grill', 'food', 'pizza', 'burger']):
                category = 'Food & Dining'
            elif any(term in merchant_lower for term in ['gas', 'fuel', 'station', 'uber', 'lyft', 'taxi', 'parking']):
                category = 'Transportation'
            elif any(term in merchant_lower for term in ['amazon', 'store', 'shop', 'retail', 'target', 'walmart', 'mall']):
                category = 'Shopping'
            elif any(term in merchant_lower for term in ['medical', 'hospital', 'pharmacy', 'clinic', 'health', 'doctor']):
                category = 'Health & Medical'
            elif any(term in merchant_lower for term in ['electric', 'utility', 'services', 'internet', 'phone', 'cable']):
                if 50 <= transaction.amount <= 500:
                    category = 'Bills & Utilities'
                else:
                    category = 'Other'
            elif any(term in merchant_lower for term in ['netflix', 'spotify', 'entertainment', 'theater', 'movie']):
                category = 'Entertainment'
            else:
                # Amount-based inference for generic names
                if transaction.amount < 20:
                    category = 'Food & Dining'  # Small amounts often food
                elif 20 <= transaction.amount <= 100:
                    category = 'Shopping'      # Medium amounts often shopping
                elif transaction.amount > 100:
                    category = 'Bills & Utilities'  # Large amounts often bills
                else:
                    category = 'Other'
            
            confidence = 0.5
            
            return CategoryPrediction(
                category=category,
                confidence_score=confidence,
                agent_name="pattern_analysis",
                reasoning=f"Pattern analysis for merchant '{clean_merchant}' [LLM web search attempted but failed]"
            )
            
        except Exception as e:
            logger.error(f"Error in web search classification: {e}")
            return None
    

    
    def _parse_web_search_response(self, response_text: str, merchant_name: str) -> Tuple[str, float, str]:
        """Parse LLM response from web search to extract category, confidence, and reasoning."""
        try:
            response_lower = response_text.lower()
            
            # Look for explicit category mentions in response
            category_found = None
            confidence = 0.7  # Default confidence for web search results
            
            # Check for category keywords in the response
            for category in self.categories:
                if category.lower() in response_lower:
                    category_found = category
                    break
            
            # If no explicit category found, try pattern matching on business types
            if not category_found:
                if any(term in response_lower for term in ['restaurant', 'food', 'dining', 'cafe', 'pizza', 'burger']):
                    category_found = 'Food & Dining'
                elif any(term in response_lower for term in ['gas', 'fuel', 'automotive', 'transportation', 'uber', 'lyft']):
                    category_found = 'Transportation'
                elif any(term in response_lower for term in ['store', 'retail', 'shopping', 'market', 'mall']):
                    category_found = 'Shopping'
                elif any(term in response_lower for term in ['entertainment', 'movie', 'theater', 'cinema']):
                    category_found = 'Entertainment'
                elif any(term in response_lower for term in ['medical', 'health', 'pharmacy', 'hospital']):
                    category_found = 'Health & Medical'
                elif any(term in response_lower for term in ['utility', 'electric', 'gas company', 'internet', 'phone']):
                    category_found = 'Bills & Utilities'
                elif any(term in response_lower for term in ['bank', 'financial', 'investment', 'fee']):
                    category_found = 'Financial'
                else:
                    # Fallback to simple pattern analysis
                    merchant_lower = merchant_name.lower()
                    if any(term in merchant_lower for term in ['cafe', 'coffee', 'restaurant', 'kitchen', 'food', 'pizza', 'burger']):
                        category_found = 'Food & Dining'
                    elif any(term in merchant_lower for term in ['gas', 'fuel', 'station', 'uber', 'lyft', 'taxi', 'parking']):
                        category_found = 'Transportation'
                    elif any(term in merchant_lower for term in ['amazon', 'store', 'shop', 'retail', 'target', 'walmart', 'mall']):
                        category_found = 'Shopping'
                    elif any(term in merchant_lower for term in ['medical', 'hospital', 'pharmacy', 'clinic', 'health', 'doctor']):
                        category_found = 'Health & Medical'
                    elif any(term in merchant_lower for term in ['electric', 'utility', 'services', 'internet', 'phone', 'cable']):
                        category_found = 'Bills & Utilities'
                    elif any(term in merchant_lower for term in ['netflix', 'spotify', 'entertainment', 'theater', 'movie']):
                        category_found = 'Entertainment'
                    else:
                        category_found = 'Other'
                    confidence = 0.5
            
            # Extract reasoning (first sentence or up to 100 chars)
            reasoning_lines = response_text.strip().split('\n')
            reasoning = reasoning_lines[0][:100] if reasoning_lines else f"Classified as {category_found} based on web search"
            
            return category_found, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Error parsing web search response: {e}")
            # Fallback to simple pattern analysis
            merchant_lower = merchant_name.lower()
            if any(term in merchant_lower for term in ['cafe', 'coffee', 'restaurant', 'kitchen', 'food', 'pizza', 'burger']):
                category = 'Food & Dining'
            elif any(term in merchant_lower for term in ['gas', 'fuel', 'station', 'uber', 'lyft', 'taxi', 'parking']):
                category = 'Transportation'
            elif any(term in merchant_lower for term in ['amazon', 'store', 'shop', 'retail', 'target', 'walmart', 'mall']):
                category = 'Shopping'
            elif any(term in merchant_lower for term in ['medical', 'hospital', 'pharmacy', 'clinic', 'health', 'doctor']):
                category = 'Health & Medical'
            elif any(term in merchant_lower for term in ['electric', 'utility', 'services', 'internet', 'phone', 'cable']):
                category = 'Bills & Utilities'
            elif any(term in merchant_lower for term in ['netflix', 'spotify', 'entertainment', 'theater', 'movie']):
                category = 'Entertainment'
            else:
                category = 'Other'
            
            return category, 0.4, f"Fallback pattern classification for {merchant_name}"


    
    def llm_classification(self, transaction: TransactionData) -> Optional[CategoryPrediction]:
        """Use LLM for intelligent classification."""
        try:
            if not hasattr(self, 'model') or not self.model:
                logger.debug("Agno OpenAI model not available, skipping LLM classification")
                return None
            
            # Create a descriptive prompt for classification
            prompt = f"""
            Classify the following transaction into one of these categories:
            {', '.join(self.categories)}
            
            Transaction details:
            - Merchant: {transaction.merchant_name}
            - Amount: ${abs(transaction.amount):.2f}
            - Description: {transaction.description}
            - Date: {transaction.transaction_date.strftime('%Y-%m-%d') if transaction.transaction_date else 'Unknown'}
            
            Please respond with only the category name that best fits this transaction.
            If you are uncertain or the merchant is unknown, choose 'Other'.
            Rate your confidence on a scale of 1-10 after the category (e.g., "Food & Dining 8").
            """
            
            # Create proper Agno Message objects
            from agno.models.message import Message
            messages = [
                Message(role="system", content="You are a financial transaction categorization expert. Respond with the category name followed by your confidence score (1-10). Be conservative with confidence for unknown merchants."),
                Message(role="user", content=prompt)
            ]
            
            # Call using Agno OpenAI model with basicAgents configuration
            response = self.model.invoke(messages, assistant_message=Message(role="assistant", content=""))
            
            # Extract the category and confidence from Agno response
            response_text = response.content.strip()
            
            # Parse response for category and confidence
            predicted_category = "Other"
            confidence_score = 0.6  # Default medium confidence
            
            # Try to parse "Category Confidence" format
            parts = response_text.split()
            if len(parts) >= 2:
                try:
                    # Last part might be confidence score
                    potential_confidence = float(parts[-1])
                    if 1 <= potential_confidence <= 10:
                        confidence_score = potential_confidence / 10.0  # Convert to 0-1 scale
                        predicted_category = ' '.join(parts[:-1])
                    else:
                        predicted_category = response_text
                except ValueError:
                    predicted_category = response_text
            else:
                predicted_category = response_text
            
            # Validate the category
            if predicted_category not in self.categories:
                # If LLM returns invalid category, try to map it or fallback
                logger.warning(f"LLM returned invalid category: {predicted_category}")
                predicted_category = "Other"
                confidence_score = 0.4  # Low confidence for invalid responses
            
            # Additional confidence adjustment for unknown merchants
            merchant_lower = transaction.merchant_name.lower()
            known_patterns = ['amazon', 'starbucks', 'mcdonalds', 'walmart', 'target', 'shell', 'exxon']
            
            if not any(pattern in merchant_lower for pattern in known_patterns):
                # Unknown merchant - reduce confidence
                confidence_score = min(0.7, confidence_score)
                logger.debug(f"Reduced confidence for unknown merchant: {transaction.merchant_name}")
            
            return CategoryPrediction(
                category=predicted_category,
                confidence_score=confidence_score,
                agent_name="llm",
                reasoning=f"LLM classified '{transaction.merchant_name}' with confidence {confidence_score:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return None
    
    def classify_transaction(self, transaction: TransactionData, 
                           known_merchants: Dict[str, str] = None,
                           similar_merchants: List[Tuple[str, float, str]] = None) -> CategoryPrediction:
        """Main classification method using multiple strategies."""
        try:
            if known_merchants is None:
                known_merchants = {}
            
            # Strategy 1: Check for exact merchant match (direct lookup)
            if transaction.merchant_name in known_merchants:
                return CategoryPrediction(
                    category=known_merchants[transaction.merchant_name],
                    confidence_score=0.95,
                    agent_name="exact_match",
                    reasoning=f"Exact match for known merchant: {transaction.merchant_name}"
                )
            
            # Case-insensitive lookup
            for known_merchant, category in known_merchants.items():
                if transaction.merchant_name.lower() == known_merchant.lower():
                    return CategoryPrediction(
                        category=category,
                        confidence_score=0.9,
                        agent_name="exact_match_case_insensitive", 
                        reasoning=f"Case-insensitive match for: {transaction.merchant_name}"
                    )
            
            # Strategy 2: Rule-based classification using patterns
            # Check merchant rules
            merchant = transaction.merchant_name.lower()
            for pattern, category in self.merchant_rules.items():
                if re.search(pattern, merchant):
                    result = CategoryPrediction(
                        category=category,
                        confidence_score=0.8,
                        agent_name="rule_based_merchant",
                        reasoning=f"Merchant pattern match: {pattern}"
                    )
                    if result.confidence_score >= 0.75:
                        logger.debug(f"Rule-based classification: {result.category}")
                        return result
            
            # Check description rules
            description = transaction.description.lower()
            for pattern, category in self.description_rules.items():
                if re.search(pattern, description):
                    result = CategoryPrediction(
                        category=category,
                        confidence_score=0.7,
                        agent_name="rule_based_description",
                        reasoning=f"Description pattern match: {pattern}"
                    )
                    if result.confidence_score >= 0.75:
                        logger.debug(f"Rule-based classification: {result.category}")
                        return result
            
            # Strategy 3: Similarity-based classification
            if similar_merchants:
                result = self.similarity_based_classification(transaction, similar_merchants)
                if result and result.confidence_score >= 0.7:
                    logger.debug(f"Similarity-based classification: {result.category}")
                    return result
            
            # Strategy 4: LLM classification
            result = self.llm_classification(transaction)
            if result and result.confidence_score >= 0.8:  # Back to original threshold
                logger.debug(f"LLM classification: {result.category}")
                return result
            
            # Strategy 5: Web search for unknown merchants (original working logic)
            web_search_attempted = False
            if not result or result.confidence_score < 0.8:
                logger.info(f"Low confidence ({result.confidence_score if result else 0:.2f}) - triggering web search for {transaction.merchant_name}")
                web_search_attempted = True
                web_result = self.web_search_classification(transaction)
                if web_result and web_result.confidence_score > (result.confidence_score if result else 0):
                    logger.debug(f"Web search enhanced classification: {web_result.category}")
                    # Mark as LLM with web search
                    web_result.agent_name = "llm_with_web_search"
                    return web_result
            
            # Return LLM result with web search indicator if web search was attempted
            if result:
                if web_search_attempted:
                    result.reasoning = f"{result.reasoning} [Web search consulted]"
                return result
            
            # Final fallback
            return CategoryPrediction(
                category="Other",
                confidence_score=0.3,
                agent_name="fallback",
                reasoning="Could not determine category with confidence"
            )
            
        except Exception as e:
            logger.error(f"Error classifying transaction {transaction.id}: {e}")
            return CategoryPrediction(
                category="Other",
                confidence_score=0.1,
                agent_name="error_fallback",
                reasoning=f"Error in classification: {e}"
            )
    
    def classify_batch(self, transactions: List[TransactionData],
                      known_merchants: Dict[str, str] = None,
                      similarity_results: Dict[str, List[Tuple[str, float, str]]] = None) -> List[CategoryPrediction]:
        """Classify a batch of transactions."""
        try:
            if known_merchants is None:
                known_merchants = {}
            
            if similarity_results is None:
                similarity_results = {}
            
            predictions = []
            
            for transaction in transactions:
                # Get similarity results for this transaction
                similar_merchants = similarity_results.get(transaction.id, [])
                
                # Classify the transaction
                prediction = self.classify_transaction(
                    transaction, known_merchants, similar_merchants
                )
                
                predictions.append(prediction)
            
            logger.info(f"Classified {len(predictions)} transactions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            return []
    
    def get_classification_stats(self, predictions: List[CategoryPrediction]) -> Dict[str, Any]:
        """Get statistics about classification results."""
        try:
            if not predictions:
                return {}
            
            # Count by category
            category_counts = {}
            method_counts = {}
            confidence_scores = []
            
            for pred in predictions:
                category_counts[pred.category] = category_counts.get(pred.category, 0) + 1
                method_counts[pred.agent_name] = method_counts.get(pred.agent_name, 0) + 1
                confidence_scores.append(pred.confidence_score)
            
            # Calculate average confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # High confidence predictions
            high_confidence = len([p for p in predictions if p.confidence_score >= 0.8])
            
            return {
                "total_predictions": len(predictions),
                "category_distribution": category_counts,
                "method_distribution": method_counts,
                "average_confidence": avg_confidence,
                "high_confidence_count": high_confidence,
                "high_confidence_rate": high_confidence / len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating classification stats: {e}")
            return {}
    
    async def run(self, transactions: List[TransactionData],
                  known_merchants: Dict[str, str] = None,
                  similarity_results: Dict[str, List[Tuple[str, float, str]]] = None,
                  **kwargs) -> Dict[str, Any]:
        """Agno agent run method."""
        try:
            # Classify all transactions
            predictions = self.classify_batch(transactions, known_merchants, similarity_results)
            
            # Get statistics
            stats = self.get_classification_stats(predictions)
            
            return {
                "status": "success",
                "predictions": predictions,
                "stats": stats,
                "message": f"Classified {len(predictions)} transactions with {stats.get('average_confidence', 0):.3f} average confidence"
            }
            
        except Exception as e:
            logger.error(f"Classifier agent error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "predictions": []
            }