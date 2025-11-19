"""Custom categories management for IntelliSpend"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from config.settings import settings

logger = logging.getLogger(__name__)

# Path to custom categories storage
CUSTOM_CATEGORIES_PATH = Path(settings.DATA_DIR) / "custom_categories.json"


def load_custom_categories() -> Dict[str, Any]:
    """
    Load custom categories from storage.
    
    Returns:
        Dict with custom categories data
    """
    if not CUSTOM_CATEGORIES_PATH.exists():
        return {
            "categories": [],
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    try:
        with open(CUSTOM_CATEGORIES_PATH, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading custom categories: {e}")
        return {
            "categories": [],
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }


def save_custom_categories(data: Dict[str, Any]) -> bool:
    """
    Save custom categories to storage.
    
    Args:
        data: Custom categories data dict
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        CUSTOM_CATEGORIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Update last_updated timestamp
        data['last_updated'] = datetime.now().isoformat()
        
        with open(CUSTOM_CATEGORIES_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data.get('categories', []))} custom categories")
        return True
    except Exception as e:
        logger.error(f"Error saving custom categories: {e}")
        return False


def get_all_custom_categories() -> List[Dict[str, Any]]:
    """
    Get all custom categories.
    
    Returns:
        List of custom category dicts
    """
    data = load_custom_categories()
    return data.get('categories', [])


def get_custom_category_by_name(category_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a custom category by name.
    
    Args:
        category_name: Name of the category
        
    Returns:
        Category dict if found, None otherwise
    """
    categories = get_all_custom_categories()
    for cat in categories:
        if cat.get('name', '').lower() == category_name.lower():
            return cat
    return None


def create_custom_category(
    name: str,
    description: str = "",
    keywords: List[str] = None,
    created_by: str = "user"
) -> Dict[str, Any]:
    """
    Create a new custom category.
    
    Args:
        name: Category name
        description: Category description
        keywords: List of keywords for matching
        created_by: Who created this category
        
    Returns:
        Result dict with success status
    """
    if keywords is None:
        keywords = []
    
    # Check if category already exists
    existing = get_custom_category_by_name(name)
    if existing:
        return {
            "success": False,
            "error": f"Category '{name}' already exists",
            "category": existing
        }
    
    # Check if it conflicts with standard taxonomy
    standard_categories = settings.get_category_names()
    if name in standard_categories:
        return {
            "success": False,
            "error": f"Category '{name}' already exists in standard taxonomy",
            "category": None
        }
    
    # Create new category
    new_category = {
        "id": f"custom_{name.lower().replace(' ', '_')}",
        "name": name,
        "description": description,
        "keywords": keywords,
        "is_custom": True,
        "created_at": datetime.now().isoformat(),
        "created_by": created_by,
        "transaction_count": 0  # Track how many transactions use this
    }
    
    # Load existing data
    data = load_custom_categories()
    if 'categories' not in data:
        data['categories'] = []
    
    # Add new category
    data['categories'].append(new_category)
    
    # Save
    if save_custom_categories(data):
        logger.info(f"Created custom category: {name}")
        return {
            "success": True,
            "category": new_category,
            "message": f"Custom category '{name}' created successfully"
        }
    else:
        return {
            "success": False,
            "error": "Failed to save custom category",
            "category": None
        }


def delete_custom_category(category_name: str) -> Dict[str, Any]:
    """
    Delete a custom category.
    
    Args:
        category_name: Name of the category to delete
        
    Returns:
        Result dict with success status and affected transactions info
    """
    data = load_custom_categories()
    categories = data.get('categories', [])
    
    # Find and remove category
    removed_category = None
    updated_categories = []
    for cat in categories:
        if cat.get('name', '').lower() == category_name.lower():
            removed_category = cat
        else:
            updated_categories.append(cat)
    
    if removed_category is None:
        return {
            "success": False,
            "error": f"Category '{category_name}' not found",
            "affected_count": 0
        }
    
    # Update data
    data['categories'] = updated_categories
    save_custom_categories(data)
    
    # Get transaction count for info
    transaction_count = removed_category.get('transaction_count', 0)
    
    logger.info(f"Deleted custom category: {category_name} (affected {transaction_count} transactions)")
    
    return {
        "success": True,
        "category": removed_category,
        "affected_count": transaction_count,
        "message": f"Custom category '{category_name}' deleted. {transaction_count} transactions need reclassification."
    }


def increment_category_usage(category_name: str) -> bool:
    """
    Increment the transaction count for a custom category.
    
    Args:
        category_name: Name of the category
        
    Returns:
        True if successful
    """
    data = load_custom_categories()
    categories = data.get('categories', [])
    
    for cat in categories:
        if cat.get('name', '').lower() == category_name.lower():
            cat['transaction_count'] = cat.get('transaction_count', 0) + 1
            save_custom_categories(data)
            return True
    
    return False


def get_all_categories_with_custom() -> List[str]:
    """
    Get all category names including custom categories.
    
    Returns:
        List of category names (standard + custom)
    """
    standard = settings.get_category_names()
    custom = [cat['name'] for cat in get_all_custom_categories()]
    return standard + custom


def is_custom_category(category_name: str) -> bool:
    """
    Check if a category is a custom category.
    
    Args:
        category_name: Name of the category
        
    Returns:
        True if custom, False otherwise
    """
    custom = get_custom_category_by_name(category_name)
    return custom is not None


def get_custom_categories_summary() -> Dict[str, Any]:
    """
    Get summary statistics about custom categories.
    
    Returns:
        Dict with summary stats
    """
    categories = get_all_custom_categories()
    total_transactions = sum(cat.get('transaction_count', 0) for cat in categories)
    
    return {
        "total_custom_categories": len(categories),
        "total_transactions_in_custom": total_transactions,
        "categories": [
            {
                "name": cat['name'],
                "description": cat.get('description', ''),
                "transaction_count": cat.get('transaction_count', 0),
                "created_at": cat.get('created_at', '')
            }
            for cat in categories
        ]
    }

