"""Preprocessor Agent for IntelliSpend - Cleans and normalizes transaction data"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from typing import Dict, Any, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

from config.settings import settings
from agents.tools import normalize_transaction, batch_normalize_transactions

logger = logging.getLogger(__name__)

# Load environment configuration
load_dotenv(Path(__file__).parent.parent / '.env')


def create_preprocessor_agent() -> Agent:
    """
    Create Preprocessor Agent with OpenAI/Azure OpenAI configuration.
    
    Supports both:
    - Azure OpenAI (requires AZURE_OPENAI_ENDPOINT, MODEL_API_VERSION)
    - Direct OpenAI (uses default OpenAI endpoint)
    
    The Preprocessor Agent is responsible for:
    - Cleaning and normalizing transaction descriptions
    - Extracting payment mode information
    - Preparing data for downstream processing
    """
    # Required environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    model_name = os.getenv('MODEL_NAME', 'gpt-4o-mini')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = os.getenv('MODEL_API_VERSION')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    
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
    
    # Create toolkit with preprocessing tools
    # Note: Agno Toolkit accepts functions directly
    try:
        toolkit = Toolkit(
            name="PreprocessingTools",
            tools=[
                normalize_transaction,
                batch_normalize_transactions
            ]
        )
        tools_list = [toolkit]
    except Exception as e:
        # Fallback: use functions directly if Toolkit doesn't work
        logger.warning(f"Toolkit creation failed: {e}. Using functions directly.")
        tools_list = [normalize_transaction, batch_normalize_transactions]
    
    # Create agent
    agent = Agent(
        name="PreprocessorAgent",
        model=model_config,
        description=(
            "I am the Preprocessor Agent for IntelliSpend. "
            "My role is to clean, normalize, and enrich transaction data. "
            "I can normalize transaction descriptions, extract payment modes, "
            "and prepare data for classification. "
            "I ensure transaction strings are standardized and ready for processing."
        ),
        tools=tools_list,
        markdown=True,
        instructions=[
            "When asked to normalize a transaction, use the normalize_transaction tool.",
            "For batch processing, use batch_normalize_transactions tool.",
            "Always return structured results with original and normalized versions.",
            "Extract payment mode information when available (UPI, NEFT, IMPS, CARD, etc.)."
        ]
    )
    
    return agent


def preprocess_transaction(description: str) -> Dict[str, Any]:
    """
    Convenience function to preprocess a single transaction.
    
    Args:
        description: Raw transaction description
        
    Returns:
        Preprocessed transaction data
    """
    agent = create_preprocessor_agent()
    
    prompt = f"""
    Normalize this transaction description: "{description}"
    
    Use the normalize_transaction tool to process it.
    """
    
    response = agent.run(prompt)
    return response.content


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the preprocessor agent
    print("🔧 Testing Preprocessor Agent...\n")
    
    try:
        agent = create_preprocessor_agent()
        
        # Test cases
        test_transactions = [
            "AMAZON PAY INDIA TXN 12345",
            "UBER TRIP MUMBAI 67890 UPI",
            "STARBUCKS COFFEE MUMBAI CARD",
            "ZOMATO ORDER 98765"
        ]
        
        for transaction in test_transactions:
            print(f"📝 Processing: {transaction}")
            prompt = f'Normalize this transaction: "{transaction}"'
            response = agent.run(prompt)
            print(f"✅ Result: {response.content}\n")
            print("-" * 80)
            print()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please check your .env configuration and ensure all dependencies are installed")

