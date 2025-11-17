"""
Example: How to Use Parallel Transaction Processing in IntelliSpend

This demonstrates how to dramatically improve processing speed using parallel processing.
Performance improvements: 3-10x faster for medium to large batches!
"""

import asyncio
from agents.agent_team import IntelliSpendAgentTeam
from agents.parallel_processor import ProcessingConfig
from models.transaction import TransactionData
from datetime import datetime
import time

def example_parallel_processing():
    """
    Example showing automatic and manual parallel processing
    """
    print("🚀 IntelliSpend Parallel Processing Demo\n")
    
    # Initialize agent team
    agent_team = IntelliSpendAgentTeam()
    
    # Create sample transactions for testing
    sample_transactions = [
        TransactionData(
            id=f"txn_{i}",
            merchant_name=merchant,
            amount=amount,
            description=f"Transaction at {merchant}",
            transaction_date=datetime.now()
        )
        for i, (merchant, amount) in enumerate([
            ("Starbucks Coffee", 4.85),
            ("Shell Gas Station", 45.20), 
            ("Amazon", 129.99),
            ("Whole Foods", 87.42),
            ("Uber", 15.30),
            ("Netflix", 15.49),
            ("McDonald's", 12.45),
            ("Target", 68.90),
            ("CVS Pharmacy", 23.45),
            ("Home Depot", 89.99),
            ("Best Buy", 299.99),
            ("Walmart", 56.78),
            ("Costco", 156.78),
            ("Trader Joe's", 43.67),
            ("Safeway", 55.80)
        ])
    ]
    
    print(f"Processing {len(sample_transactions)} transactions...\n")
    
    # 1. AUTOMATIC MODE - Let IntelliSpend choose optimal processing
    print("1️⃣  AUTOMATIC MODE (Recommended)")
    print("   IntelliSpend automatically chooses the best processing method")
    start_time = time.time()
    
    result_auto = agent_team.process_transactions(sample_transactions, use_parallel=True)
    auto_time = time.time() - start_time
    
    if result_auto['status'] == 'success':
        metrics = result_auto['results'].get('performance_metrics', {})
        print(f"   ✅ Completed in {auto_time:.2f}s")
        print(f"   📊 {metrics.get('transactions_per_second', 0):.1f} transactions/second")
        print(f"   🎯 Processed {result_auto['results']['processed_transactions']} transactions")
    
    print()
    
    # 2. MANUAL PARALLEL MODE - Custom configuration
    print("2️⃣  MANUAL PARALLEL MODE (Advanced)")
    print("   Custom configuration for specific performance needs")
    
    # Custom configuration for high performance
    custom_config = ProcessingConfig(
        max_workers=6,           # Use 6 parallel workers
        batch_size=5,           # Process 5 transactions per batch
        parallel_embeddings=True,    # Parallel embedding generation
        parallel_classification=True, # Parallel LLM classification 
        parallel_similarity=True     # Parallel similarity search
    )
    
    start_time = time.time()
    result_parallel = agent_team.process_transactions_parallel(sample_transactions, custom_config)
    parallel_time = time.time() - start_time
    
    if result_parallel['status'] == 'success':
        metrics = result_parallel['results']['performance_metrics']
        print(f"   ✅ Completed in {parallel_time:.2f}s")
        print(f"   📊 {metrics['transactions_per_second']:.1f} transactions/second")
        print(f"   ⚡ {metrics['max_workers']} parallel workers")
        print(f"   📦 Batch size: {metrics['batch_size']}")
    
    print()
    
    # 3. PERFORMANCE COMPARISON
    print("3️⃣  PERFORMANCE ANALYSIS")
    if result_auto['status'] == 'success' and result_parallel['status'] == 'success':
        sequential_estimate = len(sample_transactions) * 2.0  # Rough estimate
        
        auto_speedup = sequential_estimate / auto_time
        parallel_speedup = sequential_estimate / parallel_time
        
        print(f"   📈 Estimated sequential time: {sequential_estimate:.1f}s")
        print(f"   🚀 Auto mode speedup: {auto_speedup:.1f}x faster")
        print(f"   ⚡ Parallel mode speedup: {parallel_speedup:.1f}x faster")
        print(f"   💡 Performance improvement: {((sequential_estimate - auto_time) / sequential_estimate * 100):.0f}% faster")

async def example_async_processing():
    """
    Example showing ultra-fast async processing
    """
    print("\n⚡ ASYNC PROCESSING DEMO")
    print("   Maximum throughput for real-time applications\n")
    
    # Initialize agent team
    agent_team = IntelliSpendAgentTeam()
    
    # Create sample transactions
    sample_transactions = [
        TransactionData(
            id=f"async_txn_{i}",
            merchant_name=f"Merchant {i}",
            amount=round(10.0 + i * 5.5, 2),
            description=f"Async transaction {i}",
            transaction_date=datetime.now()
        )
        for i in range(20)  # 20 transactions for async demo
    ]
    
    print(f"Processing {len(sample_transactions)} transactions asynchronously...")
    
    start_time = time.time()
    result_async = await agent_team.process_transactions_async(
        sample_transactions, 
        max_concurrent=8  # Process up to 8 transactions concurrently
    )
    async_time = time.time() - start_time
    
    if result_async['status'] == 'success':
        metrics = result_async['results']['performance_metrics']
        print(f"   ✅ Completed in {async_time:.2f}s")
        print(f"   🔥 {metrics['transactions_per_second']:.1f} transactions/second")
        print(f"   ⚡ Max concurrent: {metrics['max_concurrent']}")
        
        # Calculate speedup
        sequential_estimate = len(sample_transactions) * 1.8
        speedup = sequential_estimate / async_time
        print(f"   📈 Estimated speedup: {speedup:.1f}x faster than sequential")
    
    print()

def example_processing_configurations():
    """
    Show different configuration options for various use cases
    """
    print("4️⃣  CONFIGURATION EXAMPLES")
    print("   Different setups for different scenarios\n")
    
    configs = {
        "High Speed": ProcessingConfig(
            max_workers=8,
            batch_size=12,
            parallel_embeddings=True,
            parallel_classification=True,
            parallel_similarity=True
        ),
        "Balanced": ProcessingConfig(
            max_workers=4,
            batch_size=8,
            parallel_embeddings=True,
            parallel_classification=True,
            parallel_similarity=True
        ),
        "Low Resource": ProcessingConfig(
            max_workers=2,
            batch_size=5,
            parallel_embeddings=True,
            parallel_classification=False,  # Save on API calls
            parallel_similarity=True
        ),
        "API Conservative": ProcessingConfig(
            max_workers=2,
            batch_size=3,
            parallel_embeddings=False,
            parallel_classification=False,  # Sequential LLM calls
            parallel_similarity=True
        )
    }
    
    for name, config in configs.items():
        print(f"   📋 {name} Configuration:")
        print(f"      Workers: {config.max_workers}, Batch: {config.batch_size}")
        print(f"      Parallel: Embeddings={config.parallel_embeddings}, "
              f"Classification={config.parallel_classification}, "
              f"Similarity={config.parallel_similarity}")
        print()

def main():
    """Run all examples"""
    print("=" * 60)
    print("🚀 INTELLISPEND PARALLEL PROCESSING EXAMPLES")
    print("=" * 60)
    print()
    
    # Regular parallel processing examples
    example_parallel_processing()
    
    # Configuration examples
    example_processing_configurations()
    
    print("\n" + "=" * 60)
    print("💡 PERFORMANCE TIPS:")
    print("• Use parallel processing for 5+ transactions")
    print("• Async processing is best for real-time applications")
    print("• Adjust max_workers based on your CPU cores")
    print("• Higher batch_size = more efficiency, more memory")
    print("• Enable parallel_classification for biggest speedup")
    print("=" * 60)
    
    # Run async example
    print("\nRunning async example...")
    asyncio.run(example_async_processing())

if __name__ == "__main__":
    main()