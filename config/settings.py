"""Configuration settings for IntelliSpend"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')


class Settings:
    """Centralized configuration management for IntelliSpend"""
    
    # OpenAI/Azure Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    MODEL_API_VERSION = os.getenv('MODEL_API_VERSION', '2024-08-01-preview')
    MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '')
    
    # Build endpoint URL for Azure OpenAI
    @property
    def MODEL_ENDPOINT(self) -> str:
        if self.AZURE_OPENAI_ENDPOINT:
            return (
                f"{self.AZURE_OPENAI_ENDPOINT}/openai/deployments/{self.MODEL_NAME}/chat/completions"
                f"?api-version={self.MODEL_API_VERSION}"
            )
        return "https://api.openai.com/v1/chat/completions"
    
    # Embedding Model
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    
    # FAISS Configuration
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'data/vector_data/merchant_rag_index.faiss')
    MERCHANT_METADATA_PATH = os.getenv('MERCHANT_METADATA_PATH', 'data/vector_data/merchant_rag_metadata.pkl')
    MERCHANT_SEED_PATH = os.getenv('MERCHANT_SEED_PATH', 'data/merchants_seed.csv')
    
    # Classification Thresholds
    LOCAL_MATCH_THRESHOLD = float(os.getenv('LOCAL_MATCH_THRESHOLD', '0.76'))
    CONFIDENCE_HIGH_THRESHOLD = float(os.getenv('CONFIDENCE_HIGH_THRESHOLD', '0.85'))
    CONFIDENCE_MEDIUM_THRESHOLD = float(os.getenv('CONFIDENCE_MEDIUM_THRESHOLD', '0.70'))
    
    # Performance Settings
    SKIP_LLM_FALLBACK = os.getenv('SKIP_LLM_FALLBACK', 'False').lower() == 'true'
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '128'))
    TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', '5'))
    
    # Taxonomy Configuration
    TAXONOMY_CONFIG_PATH = os.getenv('TAXONOMY_CONFIG_PATH', 'config/taxonomy.json')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/intellispend.log')
    
    # Taxonomy data (loaded from JSON)
    _taxonomy_cache: Dict[str, Any] = None
    
    @classmethod
    def load_taxonomy(cls) -> Dict[str, Any]:
        """Load taxonomy configuration from JSON file"""
        if cls._taxonomy_cache is None:
            taxonomy_path = Path(cls.TAXONOMY_CONFIG_PATH)
            if not taxonomy_path.exists():
                # Use default taxonomy if file doesn't exist
                cls._taxonomy_cache = {
                    "categories": [],
                    "version": "1.0",
                    "last_updated": "2025-01-27"
                }
            else:
                with open(taxonomy_path, 'r') as f:
                    cls._taxonomy_cache = json.load(f)
        return cls._taxonomy_cache
    
    @classmethod
    def get_categories(cls) -> List[Dict[str, Any]]:
        """Get list of all categories from taxonomy"""
        taxonomy = cls.load_taxonomy()
        return taxonomy.get('categories', [])
    
    @classmethod
    def get_category_names(cls) -> List[str]:
        """Get list of category names"""
        categories = cls.get_categories()
        return [cat['name'] for cat in categories]
    
    @classmethod
    def get_category_by_id(cls, category_id: str) -> Dict[str, Any]:
        """Get category by ID"""
        categories = cls.get_categories()
        for cat in categories:
            if cat.get('id') == category_id:
                return cat
        return None
    
    @classmethod
    def validate_config(cls) -> tuple[bool, List[str]]:
        """Validate configuration and return (is_valid, missing_vars)"""
        missing = []
        
        if not cls.OPENAI_API_KEY:
            missing.append('OPENAI_API_KEY')
        
        if cls.AZURE_OPENAI_ENDPOINT and not cls.MODEL_API_VERSION:
            missing.append('MODEL_API_VERSION')
        
        # Check if taxonomy file exists
        taxonomy_path = Path(cls.TAXONOMY_CONFIG_PATH)
        if not taxonomy_path.exists():
            missing.append(f'TAXONOMY_CONFIG_PATH (file not found: {cls.TAXONOMY_CONFIG_PATH})')
        
        return len(missing) == 0, missing


# Global settings instance
settings = Settings()

