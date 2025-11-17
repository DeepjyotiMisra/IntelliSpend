"""
Transaction Data Models for IntelliSpend

Defines the core data structures for financial transactions and their categorization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid


class TransactionType(Enum):
    """Types of financial transactions"""
    DEBIT = "debit"
    CREDIT = "credit"
    TRANSFER = "transfer"


class ConfidenceLevel(Enum):
    """Confidence levels for AI categorization"""
    HIGH = "high"       # 90%+ confidence
    MEDIUM = "medium"   # 70-89% confidence  
    LOW = "low"         # 50-69% confidence
    UNKNOWN = "unknown" # <50% confidence


@dataclass
class TransactionData:
    """Core transaction data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    amount: float = 0.0
    currency: str = "USD"
    merchant_name: str = ""
    description: str = ""
    transaction_date: datetime = field(default_factory=datetime.now)
    transaction_type: TransactionType = TransactionType.DEBIT
    account_id: Optional[str] = None
    reference_number: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize merchant name and description"""
        self.merchant_name = self.merchant_name.strip().title()
        self.description = self.description.strip()


@dataclass
class CategoryPrediction:
    """AI categorization prediction result"""
    category: str
    subcategory: Optional[str] = None
    confidence_score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    reasoning: str = ""
    similar_merchants: List[str] = field(default_factory=list)
    agent_name: str = ""
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Set confidence level based on score"""
        if self.confidence_score >= 0.9:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.7:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence_score >= 0.5:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.UNKNOWN


@dataclass 
class CategorizedTransaction:
    """Complete categorized transaction with AI predictions"""
    transaction: TransactionData
    primary_prediction: CategoryPrediction
    alternative_predictions: List[CategoryPrediction] = field(default_factory=list)
    manual_category: Optional[str] = None  # User override
    is_verified: bool = False
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def final_category(self) -> str:
        """Returns the final category (manual override or AI prediction)"""
        return self.manual_category or self.primary_prediction.category
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if primary prediction has high confidence"""
        return self.primary_prediction.confidence_level == ConfidenceLevel.HIGH


# Standard category mappings for financial transactions
STANDARD_CATEGORIES = {
    "food_dining": {
        "name": "Food & Dining",
        "subcategories": ["restaurants", "fast_food", "groceries", "coffee_shops", "bars_alcohol"]
    },
    "transportation": {
        "name": "Transportation", 
        "subcategories": ["gas_fuel", "public_transport", "rideshare", "parking", "auto_maintenance"]
    },
    "shopping": {
        "name": "Shopping",
        "subcategories": ["clothing", "electronics", "home_garden", "books_media", "general_merchandise"]
    },
    "entertainment": {
        "name": "Entertainment",
        "subcategories": ["movies_theater", "streaming_services", "games", "events_concerts", "hobbies"]
    },
    "utilities_bills": {
        "name": "Utilities & Bills",
        "subcategories": ["electricity", "water", "gas", "internet", "phone", "trash"]
    },
    "healthcare": {
        "name": "Healthcare",
        "subcategories": ["doctor_visits", "pharmacy", "dental", "insurance", "fitness"]
    },
    "financial": {
        "name": "Financial",
        "subcategories": ["bank_fees", "investments", "insurance", "loans", "transfers"]
    },
    "education": {
        "name": "Education", 
        "subcategories": ["tuition", "books_supplies", "online_courses", "training"]
    },
    "travel": {
        "name": "Travel",
        "subcategories": ["flights", "hotels", "car_rental", "travel_expenses"]
    },
    "income": {
        "name": "Income",
        "subcategories": ["salary", "freelance", "investments", "refunds", "other_income"]
    },
    "other": {
        "name": "Other", 
        "subcategories": ["miscellaneous", "uncategorized"]
    }
}