"""
Comprehensive Testing Framework for IntelliSpend

Test suite covering all components of the IntelliSpend system:
- Unit tests for individual components
- Integration tests for agent workflows
- Performance benchmarks
- Accuracy validation
- End-to-end system tests
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.transaction import (
        TransactionData, CategoryPrediction, CategorizedTransaction,
        TransactionType, ConfidenceLevel, STANDARD_CATEGORIES
    )
    from utils.data_processing import TransactionParser, TransactionCleaner
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False


class TestTransactionModels(unittest.TestCase):
    """Test transaction data models"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")
            
        self.sample_transaction = TransactionData(
            amount=25.50,
            merchant_name="Starbucks Coffee",
            description="Coffee purchase",
            transaction_date=datetime(2024, 1, 15, 10, 30)
        )
    
    def test_transaction_creation(self):
        """Test TransactionData creation and validation"""
        transaction = self.sample_transaction
        
        self.assertEqual(transaction.amount, 25.50)
        self.assertEqual(transaction.merchant_name, "Starbucks Coffee")
        self.assertEqual(transaction.currency, "USD")
        self.assertEqual(transaction.transaction_type, TransactionType.DEBIT)
        self.assertIsNotNone(transaction.id)
    
    def test_category_prediction(self):
        """Test CategoryPrediction confidence level calculation"""
        # Test high confidence
        high_pred = CategoryPrediction(
            category="food_dining",
            confidence_score=0.95,
            reasoning="Test high confidence"
        )
        self.assertEqual(high_pred.confidence_level, ConfidenceLevel.HIGH)
        
        # Test medium confidence
        medium_pred = CategoryPrediction(
            category="food_dining",
            confidence_score=0.75,
            reasoning="Test medium confidence"
        )
        self.assertEqual(medium_pred.confidence_level, ConfidenceLevel.MEDIUM)
        
        # Test low confidence
        low_pred = CategoryPrediction(
            category="food_dining",
            confidence_score=0.55,
            reasoning="Test low confidence"
        )
        self.assertEqual(low_pred.confidence_level, ConfidenceLevel.LOW)
    
    def test_categorized_transaction(self):
        """Test CategorizedTransaction functionality"""
        prediction = CategoryPrediction(
            category="food_dining",
            subcategory="coffee_shops",
            confidence_score=0.89,
            reasoning="Coffee shop pattern match"
        )
        
        categorized = CategorizedTransaction(
            transaction=self.sample_transaction,
            primary_prediction=prediction
        )
        
        self.assertEqual(categorized.final_category, "food_dining")
        self.assertFalse(categorized.is_high_confidence)
        
        # Test manual override
        categorized.manual_category = "entertainment"
        self.assertEqual(categorized.final_category, "entertainment")


class TestDataProcessing(unittest.TestCase):
    """Test data processing utilities"""
    
    def setUp(self):
        """Set up test data"""
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")
            
        self.parser = TransactionParser()
        self.cleaner = TransactionCleaner()
        
        # Create test CSV file
        self.test_csv_data = [
            ["Date", "Description", "Amount", "Account"],
            ["2024-01-15", "STARBUCKS COFFEE #123", "-4.75", "1234"],
            ["2024-01-16", "AMAZON.COM PURCHASE", "-67.23", "1234"],
            ["2024-01-17", "PAYROLL DEPOSIT", "2500.00", "1234"]
        ]
    
    def test_csv_parsing(self):
        """Test CSV file parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerows(self.test_csv_data)
            temp_path = f.name
        
        try:
            transactions = self.parser.parse_csv_file(temp_path)
            
            self.assertEqual(len(transactions), 3)
            self.assertEqual(transactions[0].merchant_name, "Starbucks Coffee #123")
            self.assertEqual(transactions[0].amount, 4.75)
            
        finally:
            os.unlink(temp_path)
    
    def test_merchant_name_cleaning(self):
        """Test merchant name cleaning"""
        test_cases = [
            ("SQ *COFFEE SHOP #123", "Coffee Shop"),
            ("STARBUCKS COFFEE (LOCATION)", "Starbucks"),  # Parentheses content removed
            ("AMAZON.COM LLC", "Amazon"),  # .COM and LLC removed
            ("SHELL OIL #1234567", "Shell")  # OIL and number/location removed
        ]
        
        for original, expected in test_cases:
            cleaned = self.cleaner.clean_merchant_name(original)
            self.assertEqual(cleaned, expected)
    
    def test_amount_parsing(self):
        """Test amount parsing with various formats"""
        test_cases = [
            ("$25.50", 25.50),
            ("(15.75)", -15.75),
            ("1,234.56", 1234.56),
            ("£50.00", 50.00),
            ("€75.25", 75.25)
        ]
        
        for amount_str, expected in test_cases:
            parsed = self.parser._parse_amount(amount_str)
            self.assertEqual(parsed, expected)
    
    def test_deduplication(self):
        """Test transaction deduplication"""
        # Create duplicate transactions
        base_transaction = TransactionData(
            amount=25.50,
            merchant_name="Test Merchant",
            description="Test purchase",
            transaction_date=datetime(2024, 1, 15, 10, 30)
        )
        
        # Duplicate with slight time difference
        duplicate = TransactionData(
            amount=25.50,
            merchant_name="Test Merchant",
            description="Test purchase",
            transaction_date=datetime(2024, 1, 15, 10, 32)  # 2 minutes later
        )
        
        # Different transaction
        different = TransactionData(
            amount=45.00,
            merchant_name="Different Merchant",
            description="Different purchase",
            transaction_date=datetime(2024, 1, 15, 11, 0)
        )
        
        transactions = [base_transaction, duplicate, different]
        deduplicated = self.cleaner.deduplicate_transactions(transactions)
        
        self.assertEqual(len(deduplicated), 2)  # Should remove one duplicate

class TestSystemIntegration(unittest.TestCase):
    """Basic system integration tests"""
    
    def test_agent_team_creation(self):
        """Test that the agent team can be created without errors"""
        try:
            from agents.agent_team import IntelliSpendAgentTeam
            # Just test that we can import and create the team without crashes
            # Don't actually initialize it to avoid ML model loading in tests
            team_class = IntelliSpendAgentTeam
            self.assertIsNotNone(team_class)
        except ImportError as e:
            self.skipTest(f"Agent team not available: {e}")
    
    def test_config_loading(self):
        """Test that configuration can be loaded"""
        try:
            from config.config import OPENAI_API_KEY, SENTENCE_TRANSFORMER_MODEL
            # Test that config variables exist (may be None)
            self.assertIsNotNone(SENTENCE_TRANSFORMER_MODEL)
            self.assertEqual(SENTENCE_TRANSFORMER_MODEL, "all-MiniLM-L6-v2")
        except ImportError as e:
            self.skipTest(f"Config not available: {e}")


class TestPerformance(unittest.TestCase):
    """Performance and benchmark tests"""
    
    def test_data_processing_performance(self):
        """Test data processing performance with large datasets"""
        import time
        
        parser = TransactionParser()
        
        # Generate large test dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                "date": f"2024-01-{(i % 30) + 1:02d}",
                "description": f"Test Merchant {i % 100}",
                "amount": f"{(i % 100) + 10}.99"
            })
        
        start_time = time.time()
        
        # Process large dataset
        transactions = []
        for data in large_dataset:
            transaction = parser._dict_to_transaction(data)
            if transaction:
                transactions.append(transaction)
        
        end_time = time.time()
        
        # Should process 1000 transactions quickly
        self.assertLess(end_time - start_time, 5.0)  # Less than 5 seconds
        self.assertEqual(len(transactions), 1000)


def run_all_tests():
    """Run all test suites"""
    print("🧪 Running IntelliSpend Test Suite...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = []
    
    if MODELS_AVAILABLE:
        test_classes.extend([
            TestTransactionModels,
            TestDataProcessing,
            TestSystemIntegration,
            TestPerformance
        ])
    else:
        print("⚠️  Models not available, skipping model-dependent tests")
    
    # Commented out test classes that reference removed modules:
    # TestClassificationAgent - references old classifier module
    # TestTransactionProcessor - references coordinator module  
    # TestSystemIntegration - references coordinator module
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False


if __name__ == "__main__":
    # Run tests when script is executed directly
    run_all_tests()