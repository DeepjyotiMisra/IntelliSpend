"""Retriever Agent for IntelliSpend - Retrieves similar merchants using FAISS"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from typing import Dict, Any, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

from config.settings import settings
from agents.tools import retrieve_similar_merchants, batch_retrieve_merchants

logger = logging.getLogger(__name__)

# Load environment configuration
load_dotenv(Path(__file__).parent.parent / '.env')


def create_retriever_agent() -> Agent:
    """
    Create Retriever Agent with OpenAI/Azure OpenAI configuration.
    
    Supports both:
    - Azure OpenAI (requires AZURE_OPENAI_ENDPOINT, MODEL_API_VERSION)
    - Direct OpenAI (uses default OpenAI endpoint)
    
    The Retriever Agent is responsible for:
    - Finding similar merchants using FAISS vector search
    - Retrieving relevant merchant information and categories
    - Providing similarity scores and explanations
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
    
    # Create toolkit with retrieval tools
    # Note: Agno Toolkit accepts functions directly
    try:
        toolkit = Toolkit(
            name="RetrievalTools",
            tools=[
                retrieve_similar_merchants,
                batch_retrieve_merchants
            ]
        )
        tools_list = [toolkit]
    except Exception as e:
        # Fallback: use functions directly if Toolkit doesn't work
        logger.warning(f"Toolkit creation failed: {e}. Using functions directly.")
        tools_list = [retrieve_similar_merchants, batch_retrieve_merchants]
    
    # Create agent
    agent = Agent(
        name="RetrieverAgent",
        model=model_config,
        description=(
            "I am the Retriever Agent for IntelliSpend. "
            "My role is to find similar merchants for transaction descriptions "
            "using semantic similarity search. "
            "I use FAISS vector search to retrieve the most relevant merchants "
            "based on transaction descriptions, providing similarity scores and category information. "
            "I help identify what merchant a transaction belongs to by finding similar patterns."
        ),
        tools=tools_list,
        markdown=True,
        instructions=[
            "When asked to find similar merchants, use the retrieve_similar_merchants tool.",
            "For batch processing, use batch_retrieve_merchants tool.",
            "Always provide similarity scores and explain why merchants match.",
            "Include category information from retrieved merchants.",
            "If no good matches are found (low scores), indicate this clearly.",
            "The minimum similarity threshold is typically 0.76 for good matches."
        ]
    )
    
    return agent


def retrieve_merchants(description: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Convenience function to retrieve similar merchants for a transaction.
    
    Args:
        description: Transaction description
        top_k: Number of results to return
        
    Returns:
        Retrieval results with similar merchants
    """
    agent = create_retriever_agent()
    
    prompt = f"""
    Find similar merchants for this transaction: "{description}"
    
    Use the retrieve_similar_merchants tool with top_k={top_k}.
    Explain the results and why these merchants match.
    """
    
    response = agent.run(prompt)
    return response.content


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the retriever agent
    print("🔍 Testing Retriever Agent...\n")
    
    try:
        agent = create_retriever_agent()
        
        # Test cases
        test_transactions = [
            "AMAZON PAY INDIA TXN 12345",
            "UBER TRIP MUMBAI",
            "STARBUCKS COFFEE MUMBAI",
            "SHELL PETROL PUMP"
        ]
        
        for transaction in test_transactions:
            print(f"🔎 Searching for: {transaction}")
            prompt = f'Find similar merchants for this transaction: "{transaction}"'
            response = agent.run(prompt)
            print(f"✅ Result: {response.content}\n")
            print("-" * 80)
            print()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please check:")
        print("   1. Your .env configuration")
        print("   2. FAISS index is built (run: python utils/faiss_index_builder.py)")
        print("   3. All dependencies are installed")

