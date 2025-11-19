#!/usr/bin/env python3
"""
IntelliSpend Cleanup Script (Python version)
Resets the framework to initial state by removing generated files
"""

import os
import shutil
from pathlib import Path
import sys

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_colored(text, color=Colors.NC):
    """Print colored text"""
    print(f"{color}{text}{Colors.NC}")

def count_files(pattern, directory="."):
    """Count files matching pattern"""
    count = 0
    for path in Path(directory).rglob(pattern):
        if path.is_file():
            count += 1
    return count

def remove_files(pattern, directory=".", description=""):
    """Remove files matching pattern"""
    removed = 0
    for path in Path(directory).rglob(pattern):
        if path.is_file():
            try:
                path.unlink()
                removed += 1
            except Exception as e:
                print_colored(f"   ⚠️  Could not remove {path}: {e}", Colors.YELLOW)
    return removed

def remove_dirs(pattern, directory=".", description=""):
    """Remove directories matching pattern"""
    removed = 0
    for path in Path(directory).rglob(pattern):
        if path.is_dir():
            try:
                shutil.rmtree(path)
                removed += 1
            except Exception as e:
                print_colored(f"   ⚠️  Could not remove {path}: {e}", Colors.YELLOW)
    return removed

def main():
    print("🧹 IntelliSpend Cleanup Script")
    print("==============================")
    print()
    
    # Count files to be deleted
    output_files = count_files("*.csv", "output") + count_files("*.json", "output")
    test_files = count_files("test_*.csv", "data")
    log_files = count_files("*.log", "logs")
    # Only count test scripts in root directory
    test_scripts = len(list(Path(".").glob("test_*.py")))
    cache_dirs = len(list(Path(".").rglob("__pycache__")))
    
    total_files = output_files + test_files + log_files + test_scripts
    
    print("📋 Files to be cleaned:")
    print(f"   Output files: {output_files}")
    print(f"   Test CSV files: {test_files}")
    print(f"   Log files: {log_files}")
    print(f"   Test scripts: {test_scripts}")
    print(f"   Cache directories: {cache_dirs}")
    print()
    
    if total_files == 0 and cache_dirs == 0:
        print_colored("✅ Nothing to clean! Framework is already in clean state.", Colors.GREEN)
        return 0
    
    # Check for FAISS index
    faiss_index = Path("data/vector_data/merchant_rag_index.faiss")
    faiss_metadata = Path("data/vector_data/merchant_rag_metadata.pkl")
    faiss_exists = faiss_index.exists() or faiss_metadata.exists()
    
    delete_faiss = False
    if faiss_exists:
        print_colored("⚠️  FAISS index found: data/vector_data/merchant_rag_index.faiss", Colors.YELLOW)
        print("   This is a generated file but required for the pipeline to work.")
        response = input("   Do you want to delete it? (y/n): ").strip().lower()
        delete_faiss = response == 'y'
        print()
    
    # Confirmation
    print_colored("⚠️  This will delete all generated files listed above.", Colors.YELLOW)
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print_colored("ℹ️  Cleanup cancelled.", Colors.BLUE)
        return 0
    
    print()
    print("🧹 Cleaning up...")
    
    # 1. Remove output files
    if output_files > 0:
        print_colored("   Removing output files...", Colors.BLUE)
        removed = remove_files("*.csv", "output") + remove_files("*.json", "output")
        print_colored(f"   ✅ Removed {removed} output file(s)", Colors.GREEN)
    
    # 2. Remove test CSV files
    if test_files > 0:
        print_colored("   Removing test CSV files...", Colors.BLUE)
        removed = remove_files("test_*.csv", "data")
        print_colored(f"   ✅ Removed {removed} test file(s)", Colors.GREEN)
    
    # 3. Remove log files
    if log_files > 0:
        print_colored("   Removing log files...", Colors.BLUE)
        removed = remove_files("*.log", "logs")
        print_colored(f"   ✅ Removed {removed} log file(s)", Colors.GREEN)
    
    # 4. Remove test scripts (only in root directory)
    if test_scripts > 0:
        print_colored("   Removing test scripts...", Colors.BLUE)
        removed = 0
        for path in Path(".").glob("test_*.py"):
            try:
                path.unlink()
                removed += 1
            except Exception as e:
                print_colored(f"   ⚠️  Could not remove {path}: {e}", Colors.YELLOW)
        print_colored(f"   ✅ Removed {removed} test script(s)", Colors.GREEN)
    
    # 5. Remove cache directories
    if cache_dirs > 0:
        print_colored("   Removing Python cache directories...", Colors.BLUE)
        removed = remove_dirs("__pycache__", ".")
        print_colored(f"   ✅ Removed {removed} cache directory(ies)", Colors.GREEN)
    
    # 6. Remove FAISS index if requested
    if faiss_exists and delete_faiss:
        print_colored("   Removing FAISS index...", Colors.BLUE)
        try:
            if faiss_index.exists():
                faiss_index.unlink()
            if faiss_metadata.exists():
                faiss_metadata.unlink()
            print_colored("   ✅ Removed FAISS index files", Colors.GREEN)
        except Exception as e:
            print_colored(f"   ⚠️  Could not remove FAISS index: {e}", Colors.YELLOW)
    
    # 7. Remove temporary files
    print_colored("   Removing temporary files...", Colors.BLUE)
    removed = remove_files("*.tmp", ".") + remove_files("*.temp", ".")
    print_colored(f"   ✅ Cleaned temporary files", Colors.GREEN)
    
    # 8. Ensure directory structure
    print_colored("   Ensuring directory structure...", Colors.BLUE)
    Path("output").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("data/vector_data").mkdir(parents=True, exist_ok=True)
    print_colored("   ✅ Directory structure maintained", Colors.GREEN)
    
    print()
    print_colored("✅ Cleanup completed!", Colors.GREEN)
    print()
    print("📋 Framework Status:")
    print("   ✅ Output directory: Clean")
    print("   ✅ Test files: Removed")
    print("   ✅ Log files: Removed")
    print("   ✅ Cache: Removed")
    if faiss_exists and not delete_faiss:
        print("   ⚠️  FAISS index: Kept (required for pipeline)")
        print("      To rebuild: python utils/faiss_index_builder.py")
    elif faiss_exists and delete_faiss:
        print("   ℹ️  FAISS index: Removed")
        print("      To rebuild: python utils/faiss_index_builder.py")
    else:
        print("   ℹ️  FAISS index: Not found")
        print("      To build: python utils/faiss_index_builder.py")
    print()
    print("🚀 Ready for fresh start!")
    print()
    print("💡 Next steps:")
    print("   1. Build FAISS index: python utils/faiss_index_builder.py")
    print("   2. Run pipeline: python pipeline.py")
    print("   3. Or use: ./run_e2e_test.sh")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_colored("\n\n⚠️  Cleanup interrupted by user.", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n❌ Error during cleanup: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        sys.exit(1)

