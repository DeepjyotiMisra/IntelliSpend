import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")  # Use from .env file
MODEL_API_VERSION = os.getenv("MODEL_API_VERSION", "2024-08-01-preview")

# Model aliases for compatibility
CHAT_MODEL = MODEL_NAME
EMBEDDING_MODEL = "text-embedding-3-small"

# Construct the full OpenAI endpoint URL if configuration is available
if OPENAI_ENDPOINT and MODEL_NAME:
    OPENAI_BASE_URL = f"{OPENAI_ENDPOINT}/openai/deployments/{MODEL_NAME}/chat/completions?api-version={MODEL_API_VERSION}"
else:
    OPENAI_BASE_URL = None

# Embedding Configuration
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Local embedding model

# Agent Configuration
EMBEDDING_AGENT_TIMEOUT = 30.0
CLASSIFIER_AGENT_TIMEOUT = 45.0
FEEDBACK_AGENT_TIMEOUT = 60.0

# Request Configuration
MAX_RETRIES = 3
TIMEOUT = 60.0

# Processing Configuration - Optimized for Performance
MIN_CONFIDENCE_THRESHOLD = 0.7
WEB_SEARCH_THRESHOLD = 0.6  # Slightly higher to reduce unnecessary web searches
BATCH_SIZE = 32  # Optimized batch size for better memory usage
MAX_WORKERS = 4  # Reasonable default for parallel processing
PARALLEL_THRESHOLD = 5  # Minimum transactions to trigger parallel processing