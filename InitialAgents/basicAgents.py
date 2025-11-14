from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment configuration
load_dotenv(Path(__file__).parent.parent / '.env')

# Required environment variables
required_vars = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'MODEL_API_VERSION': os.getenv('MODEL_API_VERSION'), 
    'MODEL_NAME': os.getenv('MODEL_NAME'),
    'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT')
}

# Validate configuration
missing = [k for k, v in required_vars.items() if not v]
if missing:
    raise ValueError(f"Missing environment variables: {', '.join(missing)}")

# Build endpoint URL
MODEL_ENDPOINT = (
    f"{required_vars['AZURE_OPENAI_ENDPOINT']}/openai/deployments/{required_vars['MODEL_NAME']}/chat/completions"
    f"?api-version={required_vars['MODEL_API_VERSION']}"
)

def create_intellispend_agent():
    """Create IntelliSpend agent with Azure OpenAI configuration."""
    return Agent(
        model=OpenAIChat(
            id=required_vars['MODEL_NAME'],
            base_url=MODEL_ENDPOINT,
            extra_headers={
                "Api-Key": required_vars['OPENAI_API_KEY'],
                "Content-Type": "application/json"
            },
            timeout=60.0,
            max_retries=3
        ),
        description=(
            f"IntelliSpend AI Assistant powered by Azure OpenAI ({required_vars['MODEL_NAME']}). "
            "I help with financial analysis, spending insights, budgeting advice, "
            "and can search for financial information when needed."
        ),
        tools=[DuckDuckGoTools()],
        markdown=True
    )

def demo():
    """Run demo queries with the IntelliSpend agent."""
    print("🚀 IntelliSpend AI Assistant Demo\n")
    
    agent = create_intellispend_agent()
    
    queries = [
        "What are some effective budgeting strategies for 2025?",
        "How should I categorize my expenses for better financial tracking?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"📊 Query {i}: {query}")
        agent.print_response(query)
        if i < len(queries):
            print("-" * 80)

if __name__ == "__main__":
    try:
        agent = create_intellispend_agent()
        agent.print_response("What are the latest trends in personal finance management?")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please check your .env configuration")