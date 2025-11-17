"""
IntelliSpend - Main Entry Point

AI-powered financial transaction categorization system using agentic workflows.

Usage:
    python main.py --demo                    # Run with demo data
    python main.py --web                     # Launch web interface  
    python main.py --test                    # Run test suite
    python main.py --file <path>             # Process file
    python main.py --batch <dir>             # Process directory of files
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intellispend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from models.transaction import TransactionData, STANDARD_CATEGORIES
from agents.agent_team import IntelliSpendAgentTeam
from utils.data_processing import TransactionParser, load_sample_transactions
from utils.vector_store import MerchantVectorStore, initialize_sample_merchants


def print_banner():
    """Print IntelliSpend banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    🧠 IntelliSpend - AI Transaction Categorization v2.0       ║
    ║                                                               ║
    ║    Automated Financial Transaction Classification using        ║
    ║    Multi-Agent AI Workflows and Vector Similarity Search     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_ENDPOINT", "OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Please set up your .env file with company OpenAI credentials")
        return False
    
    return True


def run_demo():
    """Run demonstration with sample data"""
    print("🎮 Running IntelliSpend Demo...")
    print("=" * 60)
    
    if not check_environment():
        return
    
    try:
        # Create agent team
        logger.info("Initializing IntelliSpend agent team...")
        agent_team = IntelliSpendAgentTeam()
        
        # Load sample data
        logger.info("Loading sample transactions...")
        transactions = load_sample_transactions()
        print(f"📊 Loaded {len(transactions)} sample transactions")
        
        # Process transactions
        logger.info("Processing transactions...")
        processing_result = agent_team.process_transactions(transactions)
        
        if processing_result.get("status") != "success":
            print(f"❌ Processing failed: {processing_result.get('error', 'Unknown error')}")
            return
        
        results = processing_result["results"]
        predictions = results.get("predictions", [])
        
        # Display results
        print("\n📋 Processing Results:")
        print("-" * 60)
        
        for i, prediction in enumerate(predictions, 1):
            transaction = transactions[i-1]  # Get corresponding transaction
            confidence_emoji = "🟢" if prediction.confidence_score >= 0.8 else "🟡" if prediction.confidence_score >= 0.6 else "🔴"
            
            print(f"{i:2d}. {transaction.merchant_name:<20} | "
                  f"${transaction.amount:>8.2f} | "
                  f"{prediction.category:<15} | "
                  f"{confidence_emoji} {prediction.confidence_score:>6.1%}")
        
        # Generate summary from results
        total_transactions = results["processed_transactions"]
        high_confidence_count = results["pipeline_stats"]["high_confidence_predictions"]
        avg_confidence = results["pipeline_stats"]["average_confidence"]
        
        print(f"\n📈 Summary Report:")
        print("-" * 60)
        print(f"Total Processed: {total_transactions}")
        print(f"High Confidence: {high_confidence_count} ({high_confidence_count/total_transactions*100:.1f}%)")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Success Rate: {high_confidence_count/total_transactions*100:.1f}%")
        
        print(f"\n💰 Category Distribution:")
        category_counts = {}
        for prediction in predictions:
            category = prediction.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in category_counts.items():
            category_name = STANDARD_CATEGORIES.get(category, {}).get('name', category)
            print(f"  {category_name:<20}: {count:>3d} transactions")
        
        # Save results (replace previous results)
        output_file = "intellispend_demo_results.json"
        
        with open(output_file, 'w') as f:
            import json
            # Convert predictions to serializable format
            serializable_predictions = []
            for prediction in predictions:
                serializable_predictions.append({
                    "category": prediction.category,
                    "confidence_score": prediction.confidence_score,
                    "reasoning": prediction.reasoning
                })
            json.dump(serializable_predictions, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


def process_file(file_path: str):
    """Process a single file"""
    print(f"📄 Processing file: {file_path}")
    
    if not check_environment():
        return
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Create agent team and parser
        agent_team = IntelliSpendAgentTeam()
        parser = TransactionParser()
        
        # Parse file based on extension
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            transactions = parser.parse_csv_file(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            transactions = parser.parse_excel_file(file_path)
        elif file_ext == '.json':
            transactions = parser.parse_json_file(file_path)
        else:
            print(f"❌ Unsupported file format: {file_ext}")
            return
        
        if not transactions:
            print("❌ No transactions found in file")
            return
        
        print(f"📊 Parsed {len(transactions)} transactions")
        
        # Process transactions
        processing_result = agent_team.process_transactions(transactions)
        
        if processing_result.get("status") != "success":
            print(f"❌ Processing failed: {processing_result.get('error', 'Unknown error')}")
            return
        
        results = processing_result["results"]
        predictions = results.get("predictions", [])
        
        # Export results
        output_name = f"{Path(file_path).stem}_categorized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_name, 'w') as f:
            import json
            # Convert predictions to serializable format
            serializable_predictions = []
            for prediction in predictions:
                serializable_predictions.append({
                    "category": prediction.category,
                    "confidence_score": prediction.confidence_score,
                    "reasoning": prediction.reasoning
                })
            json.dump(serializable_predictions, f, indent=2)
        
        print(f"✅ Processing completed!")
        print(f"💾 Results saved to: {output_name}")
        
        # Quick summary
        total_transactions = results["processed_transactions"]
        high_confidence_count = results["pipeline_stats"]["high_confidence_predictions"]
        success_rate = (high_confidence_count / total_transactions * 100) if total_transactions > 0 else 0
        print(f"📈 Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        print(f"❌ Processing failed: {e}")


def process_batch(directory: str):
    """Process all supported files in a directory"""
    print(f"📁 Processing directory: {directory}")
    
    if not check_environment():
        return
    
    if not os.path.isdir(directory):
        print(f"❌ Directory not found: {directory}")
        return
    
    # Find supported files
    supported_extensions = ['.csv', '.xlsx', '.xls', '.json']
    files_to_process = []
    
    for ext in supported_extensions:
        files_to_process.extend(Path(directory).glob(f"*{ext}"))
    
    if not files_to_process:
        print(f"❌ No supported files found in {directory}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return
    
    print(f"📊 Found {len(files_to_process)} files to process")
    
    # Process each file
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Processing: {file_path.name}")
        process_file(str(file_path))
    
    print(f"\n✅ Batch processing completed! Processed {len(files_to_process)} files")


def run_tests():
    """Run the test suite"""
    print("🧪 Running IntelliSpend Test Suite...")
    
    try:
        from tests.test_suite import run_all_tests
        success = run_all_tests()
        
        if success:
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except ImportError as e:
        logger.error(f"Could not import test suite: {e}")
        print("❌ Test suite not available")
        return False
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")
        return False


def launch_web_interface():
    """Launch the Streamlit web interface"""
    print("🌐 Launching IntelliSpend Web Interface...")
    
    try:
        import streamlit
        import subprocess
        import sys
        
        # Launch Streamlit app
        web_app_path = "web/app.py"
        if os.path.exists(web_app_path):
            subprocess.run([sys.executable, "-m", "streamlit", "run", web_app_path])
        else:
            print(f"❌ Web app not found at: {web_app_path}")
            
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
    except Exception as e:
        logger.error(f"Web interface launch failed: {e}")
        print(f"❌ Web interface launch failed: {e}")


def setup_vector_store():
    """Initialize vector store with sample merchants"""
    print("🔧 Setting up merchant vector store...")
    
    try:
        vector_store = MerchantVectorStore()
        
        # Initialize with sample data if empty
        if len(vector_store.merchant_data) == 0:
            initialize_sample_merchants(vector_store)
            print(f"✅ Initialized vector store with {len(vector_store.merchant_data)} sample merchants")
        else:
            print(f"✅ Vector store already contains {len(vector_store.merchant_data)} merchants")
            
    except Exception as e:
        logger.error(f"Vector store setup failed: {e}")
        print(f"❌ Vector store setup failed: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="IntelliSpend - AI-Powered Transaction Categorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Run demonstration
  python main.py --web                     # Launch web interface
  python main.py --test                    # Run tests
  python main.py --file transactions.csv   # Process single file
  python main.py --batch ./data/           # Process directory
  python main.py --setup                   # Initialize vector store
        """
    )
    
    parser.add_argument("--demo", action="store_true", help="Run demo with sample data")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--file", type=str, help="Process single file")
    parser.add_argument("--batch", type=str, help="Process directory of files")
    parser.add_argument("--setup", action="store_true", help="Initialize vector store")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print_banner()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not required
    
    # Handle arguments
    if args.demo:
        run_demo()
    elif args.web:
        launch_web_interface()
    elif args.test:
        run_tests()
    elif args.file:
        process_file(args.file)
    elif args.batch:
        process_batch(args.batch)
    elif args.setup:
        setup_vector_store()
    else:
        # No arguments provided, show help and run demo
        parser.print_help()
        print("\n" + "="*60)
        print("No arguments provided. Running demo...")
        print("="*60)
        run_demo()


if __name__ == "__main__":
    main()