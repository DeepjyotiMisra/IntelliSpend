"""Feedback Agent for IntelliSpend - Processes user feedback and updates knowledge base"""

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
from agents.tools import store_feedback, get_feedback_statistics, get_taxonomy_categories

logger = logging.getLogger(__name__)

# Load environment configuration
load_dotenv(Path(__file__).parent.parent / '.env')

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


def create_feedback_agent() -> Agent:
    """
    Create Feedback Agent with configurable LLM provider (OpenAI or Google Gemini).
    
    Configuration via environment variables:
    - FEEDBACK_MODEL_PROVIDER: 'openai' or 'gemini' (default: 'gemini')
    
    For OpenAI:
    - OPENAI_API_KEY: Required
    - MODEL_NAME: Model name (default: 'gpt-4o-mini')
    - AZURE_OPENAI_ENDPOINT: Optional (for Azure OpenAI)
    - MODEL_API_VERSION: Optional (for Azure OpenAI)
    
    For Gemini:
    - GOOGLE_API_KEY: Required
    - GEMINI_MODEL_NAME: Model name (default: 'gemini-2.0-flash')
    
    The Feedback Agent is responsible for:
    - Processing user corrections and feedback
    - Analyzing feedback patterns
    - Suggesting improvements to merchant seed
    - Validating feedback quality
    - Generating insights from feedback data
    """
    provider = os.getenv('FEEDBACK_MODEL_PROVIDER', 'gemini').lower()
    
    if provider == 'gemini':
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
        OpenAIChat = _import_openai()  # Lazy import
        api_key = os.getenv('OPENAI_API_KEY')
        model_name = os.getenv('MODEL_NAME', 'gpt-4o-mini')
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = os.getenv('MODEL_API_VERSION')
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI")
        if azure_endpoint and api_version and 'your-azure' not in azure_endpoint.lower():
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
            logger.info("Using direct OpenAI configuration")
            model_config = OpenAIChat(
                id=model_name,
                api_key=api_key,
                timeout=60.0,
                max_retries=3
            )
    else:
        raise ValueError(
            f"Invalid FEEDBACK_MODEL_PROVIDER: {provider}. "
            "Must be 'openai' or 'gemini'"
        )
    
    toolkit = Toolkit(
        name="FeedbackTools",
        tools=[
            store_feedback,
            get_feedback_statistics,
            get_taxonomy_categories
        ]
    )
    tools_list = [toolkit]
    
    agent = Agent(
        name="FeedbackAgent",
        model=model_config,
        description=(
            "I am the Feedback Agent for IntelliSpend. "
            "My role is to process user feedback and corrections to improve the classification system. "
            "I analyze feedback patterns, validate corrections, and help update the merchant seed data. "
            "I provide insights on common mistakes and suggest improvements to the knowledge base. "
            "I ensure feedback is properly stored and can be used to improve future classifications."
        ),
        tools=tools_list,
        markdown=True,
        instructions=[
            "When processing feedback:",
            "1. Validate that the correction makes sense (merchant and category are reasonable)",
            "2. Use store_feedback tool to save the correction",
            "3. Analyze feedback patterns using get_feedback_statistics",
            "4. Check available categories using get_taxonomy_categories",
            "5. Provide insights on what the feedback reveals about system weaknesses",
            "",
            "For merchant corrections:",
            "- Ensure the corrected merchant name is clear and standardized",
            "- Check if this is a new merchant or a variation of an existing one",
            "- Consider if multiple transaction patterns should be added for this merchant",
            "",
            "For category corrections:",
            "- Verify the category exists in the taxonomy",
            "- Check if the correction reveals a systematic misclassification",
            "- Consider if the merchant-category mapping needs updating",
            "",
            "Always provide:",
            "- Confirmation that feedback was stored",
            "- Analysis of the feedback pattern",
            "- Suggestions for improving the system based on this feedback"
        ]
    )
    return agent


def process_feedback(
    transaction_description: str,
    original_merchant: str,
    original_category: str,
    corrected_merchant: str,
    corrected_category: str,
    amount: float = None,
    date: str = None,
    confidence_score: float = None
) -> str:
    """
    Process user feedback using the Feedback Agent.
    
    Args:
        transaction_description: Original transaction description
        original_merchant: Merchant that was originally classified
        original_category: Category that was originally assigned
        corrected_merchant: Correct merchant name (user correction)
        corrected_category: Correct category (user correction)
        amount: Transaction amount (optional)
        date: Transaction date (optional)
        confidence_score: Original confidence score (optional)
        
    Returns:
        Agent response with feedback processing results
    """
    try:
        agent = create_feedback_agent()
        
        prompt = f"""Process this user feedback:

Transaction: {transaction_description}
Original Classification: {original_merchant} / {original_category}
Corrected Classification: {corrected_merchant} / {corrected_category}
Amount: {amount if amount else 'N/A'}
Date: {date if date else 'N/A'}
Original Confidence: {confidence_score if confidence_score else 'N/A'}

Please:
1. Store this feedback using the store_feedback tool
2. Analyze the feedback pattern
3. Provide insights on what this correction reveals
4. Suggest improvements to the system based on this feedback
"""
        
        response = agent.run(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return f"Error processing feedback: {str(e)}"

