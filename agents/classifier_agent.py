"""Classifier Agent for IntelliSpend - Uses LLM reasoning to classify transactions"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agno.agent import Agent
from agno.tools import Toolkit
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging

from config.settings import settings
from agents.tools import retrieve_similar_merchants, get_taxonomy_categories
import re
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Load environment configuration
load_dotenv(Path(__file__).parent.parent / '.env')


def _load_merchant_rules() -> Dict[str, str]:
    """Load merchant-to-category mapping rules using regex patterns."""
    return {
        # Food & Dining
        r'(?i)(mcdonald\'?s|mcdonalds|burger king|kfc|taco bell|subway|pizza|restaurant|cafe|starbucks|dunkin|zomato|swiggy|food|dining|biryani|darbar|kitchen|grill|deli)': 'Food & Dining',
        r'(?i)(grocery|supermarket|walmart|target|safeway|kroger|whole foods|bigbasket|grofers|dmart|reliance fresh)': 'Food & Dining',
        
        # Transportation
        r'(?i)(uber|lyft|ola|rapido|taxi|gas|shell|exxon|bp|chevron|mobil|parking|fuel|petrol|diesel|transport)': 'Transportation',
        r'(?i)(airline|airport|flight|car rental|train|bus|metro|railway|irctc|makemytrip|goibibo)': 'Transportation',
        
        # Shopping
        r'(?i)(amazon|flipkart|ajio|myntra|ebay|shop|store|retail|mall|clothing|fashion|apparel|shopping)': 'Shopping',
        
        # Entertainment
        r'(?i)(netflix|spotify|hotstar|prime|disney|movie|theater|game|music|entertainment|streaming|cinema)': 'Entertainment',
        
        # Bills & Utilities
        r'(?i)(electric|gas company|water|internet|phone|cable|utility|bsnl|airtel|jio|vi|vodafone|reliance|broadband)': 'Bills & Utilities',
        
        # Financial
        r'(?i)(bank|atm|fee|interest|payment|transfer|loan|sbi|hdfc|icici|axis|kotak|upi|neft|imps|investment|mutual fund)': 'Financial',
        
        # Health & Medical
        r'(?i)(hospital|doctor|medical|pharmacy|health|dental|apollo|fortis|max|pharmacy|chemist|medicine|clinic)': 'Health & Medical',
        
        # Home & Garden
        r'(?i)(home depot|lowes|hardware|furniture|garden|repair|ikea|pepperfry|urban ladder|home improvement)': 'Home & Garden',
        
        # Travel
        r'(?i)(hotel|booking|travel|trip|vacation|resort|airbnb|oyo|makemytrip|goibibo|travel)': 'Travel',
        
        # Education
        r'(?i)(school|university|college|tuition|education|course|training|learning|edtech|byju|unacademy)': 'Education'
    }


def _load_description_rules() -> Dict[str, str]:
    """Load description pattern rules."""
    return {
        r'(?i)(coffee|lunch|dinner|breakfast|meal|food|snack|beverage|drink)': 'Food & Dining',
        r'(?i)(gas|fuel|gasoline|petrol|diesel|refuel)': 'Transportation',
        r'(?i)(subscription|monthly|recurring|membership|premium)': 'Bills & Utilities',
        r'(?i)(withdrawal|deposit|transfer|payment|transaction)': 'Financial',
        r'(?i)(fee|charge|penalty|service charge)': 'Financial',
        r'(?i)(refund|return|credit|cashback|reward)': 'Other',
        r'(?i)(shopping|purchase|buy|order|delivery)': 'Shopping',
        r'(?i)(movie|cinema|theater|ticket|show)': 'Entertainment'
    }


def _load_amount_rules() -> List[Tuple[float, float, str, float]]:
    """Load amount-based classification rules."""
    return [
        # (min_amount, max_amount, category, confidence)
        (0.99, 15.99, 'Food & Dining', 0.6),  # Small food purchases
        (200, 1000, 'Shopping', 0.5),         # Medium shopping
        (1000, float('inf'), 'Major Purchase', 0.4),  # Large purchases
        (0.01, 5.00, 'Fees', 0.7),           # Small fees
    ]


def apply_rule_based_classification(description: str, amount: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Apply rule-based pattern classification before LLM.
    
    Classification order: Similarity → Rule-based → LLM
    
    Args:
        description: Transaction description
        amount: Transaction amount (optional)
        
    Returns:
        Classification result if pattern matches, None otherwise
    """
    try:
        merchant_rules = _load_merchant_rules()
        description_rules = _load_description_rules()
        amount_rules = _load_amount_rules()
        
        description_lower = description.lower()
        
        # Strategy 1: Check merchant name patterns
        for pattern, category in merchant_rules.items():
            if re.search(pattern, description_lower):
                logger.debug(f"Rule-based merchant pattern match: {pattern} -> {category}")
                return {
                    "merchant": description.upper()[:50],  # Extract merchant from description
                    "category": category,
                    "confidence": "high",
                    "reasoning": f"Matched merchant pattern: {pattern}",
                    "method": "rule_based_merchant"
                }
        
        # Strategy 2: Check description patterns
        for pattern, category in description_rules.items():
            if re.search(pattern, description_lower):
                logger.debug(f"Rule-based description pattern match: {pattern} -> {category}")
                return {
                    "merchant": description.upper()[:50],
                    "category": category,
                    "confidence": "medium",
                    "reasoning": f"Matched description pattern: {pattern}",
                    "method": "rule_based_description"
                }
        
        # Strategy 3: Check amount-based rules (if amount provided)
        if amount is not None:
            for min_amt, max_amt, category, conf in amount_rules:
                if min_amt <= abs(amount) <= max_amt:
                    logger.debug(f"Rule-based amount match: {amount} -> {category}")
                    return {
                        "merchant": description.upper()[:50],
                        "category": category,
                        "confidence": "medium" if conf >= 0.6 else "low",
                        "reasoning": f"Matched amount range: ${min_amt}-${max_amt}",
                        "method": "rule_based_amount"
                    }
        
        return None  # No rule match found
        
    except Exception as e:
        logger.error(f"Error in rule-based classification: {e}")
        return None

# Lazy imports for LLM models (only import when needed)
def _import_openai():
    """Lazy import OpenAI model"""
    try:
        from agno.models.openai import OpenAIChat
        return OpenAIChat
    except ImportError as e:
        raise ImportError(f"OpenAI model not available: {e}. Install with: pip install agno[openai]")

def _import_gemini():
    """Lazy import Gemini model"""
    try:
        from agno.models.google import Gemini
        return Gemini
    except ImportError as e:
        raise ImportError(f"Gemini model not available: {e}. Install with: pip install google-genai")


def create_classifier_agent() -> Agent:
    """
    Create Classifier Agent with configurable LLM provider (OpenAI or Google Gemini).
    
    Configuration via environment variables:
    - CLASSIFIER_MODEL_PROVIDER: 'openai' or 'gemini' (default: 'gemini')
    
    For OpenAI:
    - OPENAI_API_KEY: Required
    - MODEL_NAME: Model name (default: 'gpt-4o-mini')
    - AZURE_OPENAI_ENDPOINT: Optional (for Azure OpenAI)
    - MODEL_API_VERSION: Optional (for Azure OpenAI)
    
    For Gemini:
    - GOOGLE_API_KEY: Required
    - GEMINI_MODEL_NAME: Model name (default: 'gemini-2.0-flash')
    
    The Classifier Agent is responsible for:
    - Analyzing retrieved merchants and their similarity scores
    - Using LLM reasoning to determine the best category
    - Handling ambiguous cases and edge cases
    - Providing explanations for classification decisions
    - Considering multiple factors: amount, payment mode, context
    """
    # Determine which provider to use
    provider = os.getenv('CLASSIFIER_MODEL_PROVIDER', 'gemini').lower()
    
    if provider == 'gemini':
        # Google Gemini configuration
        Gemini = _import_gemini()  # Lazy import
        api_key = os.getenv('GOOGLE_API_KEY')
        model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash')
        
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for Gemini. "
                "Get your API key from: https://aistudio.google.com/apikey"
            )
        
        logger.info(f"Using Google Gemini configuration with model: {model_name}")
        model_config = Gemini(
            id=model_name,
            api_key=api_key
        )
        
    elif provider == 'openai':
        # OpenAI configuration (supports both Azure and direct OpenAI)
        OpenAIChat = _import_openai()  # Lazy import
        api_key = os.getenv('OPENAI_API_KEY')
        model_name = os.getenv('MODEL_NAME', 'gpt-4o-mini')
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = os.getenv('MODEL_API_VERSION')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI")
        
        # Determine if using Azure or direct OpenAI
        if azure_endpoint and api_version and 'your-azure' not in azure_endpoint.lower():
            # Azure OpenAI configuration
            logger.info("Using Azure OpenAI configuration")
            model_endpoint = (
                f"{azure_endpoint}/openai/deployments/{model_name}/chat/completions"
                f"?api-version={api_version}"
            )
            model_config = OpenAIChat(
                id=model_name,
                base_url=model_endpoint,
                extra_headers={
                    "Api-Key": api_key,
                    "Content-Type": "application/json"
                },
                timeout=60.0,
                max_retries=3
            )
        else:
            # Direct OpenAI configuration
            logger.info("Using direct OpenAI configuration")
            model_config = OpenAIChat(
                id=model_name,
                api_key=api_key,
                timeout=60.0,
                max_retries=3
            )
    else:
        raise ValueError(
            f"Invalid CLASSIFIER_MODEL_PROVIDER: {provider}. "
            "Must be 'openai' or 'gemini'"
        )
    
    # Create toolkit with classification tools
    try:
        toolkit = Toolkit(
            name="ClassificationTools",
            tools=[
                retrieve_similar_merchants,
                get_taxonomy_categories
            ]
        )
        tools_list = [toolkit]
    except Exception as e:
        # Fallback: use functions directly if Toolkit doesn't work
        logger.warning(f"Toolkit creation failed: {e}. Using functions directly.")
        tools_list = [retrieve_similar_merchants, get_taxonomy_categories]
    
    # Create agent
    agent = Agent(
        name="ClassifierAgent",
        model=model_config,
        description=(
            "I am the Classifier Agent for IntelliSpend. "
            "My role is to classify transactions into categories using LLM reasoning. "
            "I analyze retrieved merchants, their similarity scores, and transaction context "
            "to determine the most appropriate category. "
            "I handle ambiguous cases, edge cases, and provide explanations for my decisions. "
            "I consider multiple factors: transaction description, amount, payment mode, "
            "and similarity scores from the retrieval system."
        ),
        tools=tools_list,
        markdown=True,
        instructions=[
            "When asked to classify a transaction:",
            "1. First, use retrieve_similar_merchants tool to find similar merchants",
            "2. Use get_taxonomy_categories tool to see available categories",
            "3. Analyze the retrieved merchants and their similarity scores",
            "4. Consider the transaction context (description, amount, payment mode)",
            "5. Use LLM reasoning to determine the best category",
            "6. Provide a clear explanation for your classification decision",
            "",
            "For high-confidence matches (score ≥ 0.76), use the best match's category.",
            "For low-confidence matches (score < 0.76), analyze all retrieved merchants and use reasoning.",
            "For ambiguous cases, consider transaction amount, payment mode, and context.",
            "Always explain your reasoning, especially for edge cases.",
            "",
            "Return structured results with:",
            "- merchant: Identified merchant name",
            "- category: Assigned category",
            "- confidence: Your confidence level (high/medium/low)",
            "- reasoning: Explanation of your decision"
        ]
    )
    
    return agent


def classify_transaction(description: str, 
                        amount: Optional[float] = None,
                        payment_mode: Optional[str] = None,
                        retrieved_merchants: Optional[list] = None) -> Dict[str, Any]:
    """
    Convenience function to classify a transaction using the Classifier Agent.
    
    Args:
        description: Transaction description
        amount: Transaction amount (optional)
        payment_mode: Payment mode (optional)
        retrieved_merchants: Pre-retrieved merchants (optional, will retrieve if not provided)
        
    Returns:
        Classification result with merchant, category, confidence, and reasoning
    """
    agent = create_classifier_agent()
    
    # Build prompt
    prompt_parts = [
        f"Classify this transaction: \"{description}\""
    ]
    
    if amount:
        prompt_parts.append(f"Amount: {amount}")
    if payment_mode:
        prompt_parts.append(f"Payment Mode: {payment_mode}")
    
    if retrieved_merchants:
        prompt_parts.append("\nRetrieved merchants (for reference only, similarity scores are low):")
        for i, merchant in enumerate(retrieved_merchants[:5], 1):
            prompt_parts.append(
                f"{i}. {merchant.get('merchant', 'Unknown')} - "
                f"{merchant.get('category', 'Unknown')} "
                f"(similarity: {merchant.get('score', 0):.3f})"
            )
        prompt_parts.append("\nNOTE: These retrieved merchants have low similarity scores and may not be relevant.")
    else:
        prompt_parts.append("\nNo similar merchants found in the database.")
    
    prompt_parts.append("\nIMPORTANT INSTRUCTIONS:")
    prompt_parts.append("1. Extract the merchant name DIRECTLY from the transaction description")
    prompt_parts.append("   - Example: 'Punjabi darbar dinner' → merchant: 'PUNJABI DARBAR'")
    prompt_parts.append("   - Do NOT use merchant names from retrieved merchants if they don't match")
    prompt_parts.append("2. Determine the category based on the transaction description and amount")
    prompt_parts.append("3. CONFIDENCE LEVEL GUIDELINES:")
    prompt_parts.append("   - Use 'high' confidence when:")
    prompt_parts.append("     * The description clearly indicates the merchant (e.g., 'AJIO SHOPPING', 'ZOMATO DELHI')")
    prompt_parts.append("     * The category is explicitly mentioned or obvious (e.g., 'SHOPPING', 'GROCERIES', 'FOOD')")
    prompt_parts.append("     * The merchant is well-known and the category matches (e.g., AMAZON → Shopping, UBER → Transport)")
    prompt_parts.append("   - Use 'medium' confidence when:")
    prompt_parts.append("     * The merchant is identifiable but category requires some inference")
    prompt_parts.append("     * The description is somewhat ambiguous but still classifiable")
    prompt_parts.append("   - Use 'low' confidence ONLY when:")
    prompt_parts.append("     * The transaction description is very vague or unclear")
    prompt_parts.append("     * You are genuinely uncertain about merchant or category")
    prompt_parts.append("     * The description doesn't provide enough information")
    prompt_parts.append("4. Provide your response in this EXACT format (no markdown, no bullets):")
    prompt_parts.append("   Merchant: [merchant name]")
    prompt_parts.append("   Category: [category name]")
    prompt_parts.append("   Confidence: [high/medium/low]")
    prompt_parts.append("   Reasoning: [brief explanation]")
    
    prompt = "\n".join(prompt_parts)
    
    response = agent.run(prompt)
    return response.content


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the classifier agent
    print("🎯 Testing Classifier Agent...\n")
    
    try:
        agent = create_classifier_agent()
        
        # Test cases
        test_transactions = [
            {
                "description": "AMAZON PAY INDIA TXN 12345",
                "amount": 1500.0,
                "payment_mode": "UPI"
            },
            {
                "description": "UNKNOWN MERCHANT XYZ 98765",
                "amount": 500.0,
                "payment_mode": "CARD"
            },
            {
                "description": "PAYMENT TO MERCHANT 12345",
                "amount": 2000.0,
                "payment_mode": "NEFT"
            }
        ]
        
        for i, transaction in enumerate(test_transactions, 1):
            print(f"📝 Test {i}: {transaction['description']}")
            print(f"   Amount: {transaction.get('amount')}, Payment: {transaction.get('payment_mode')}")
            print("   Classifying...\n")
            
            result = classify_transaction(
                transaction['description'],
                transaction.get('amount'),
                transaction.get('payment_mode')
            )
            
            print(f"✅ Result:\n{result}\n")
            print("-" * 80)
            print()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("💡 Please check:")
        provider = os.getenv('CLASSIFIER_MODEL_PROVIDER', 'gemini').lower()
        if provider == 'gemini':
            print("   1. Your .env configuration (GOOGLE_API_KEY is required)")
            print("   2. Get your Google API key from: https://aistudio.google.com/apikey")
        else:
            print("   1. Your .env configuration (OPENAI_API_KEY is required)")
            print("   2. Set CLASSIFIER_MODEL_PROVIDER=openai in .env")
        print("   3. FAISS index is built (run: python utils/faiss_index_builder.py)")
        print("   4. All dependencies are installed (run: pip install -r requirements.txt)")

