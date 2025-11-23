#!/bin/bash

# IntelliSpend Cleanup Script
# Resets the framework to initial state by removing generated files

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🧹 IntelliSpend Cleanup Script"
echo "=============================="
echo

# Count files to be deleted
OUTPUT_FILES=$(find output -name "*.csv" -o -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
TEST_FILES=$(find data -name "test_*.csv" 2>/dev/null | wc -l | tr -d ' ')
LOG_FILES=$(find logs -name "*.log" 2>/dev/null | wc -l | tr -d ' ')
TEST_SCRIPTS=$(find . -maxdepth 1 -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
CACHE_DIRS=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l | tr -d ' ')

TOTAL_FILES=$((OUTPUT_FILES + TEST_FILES + LOG_FILES + TEST_SCRIPTS))

echo "📋 Files to be cleaned:"
echo "   Output files: $OUTPUT_FILES"
echo "   Test CSV files: $TEST_FILES"
echo "   Log files: $LOG_FILES"
echo "   Test scripts: $TEST_SCRIPTS"
echo "   Cache directories: $CACHE_DIRS"
echo

if [ "$TOTAL_FILES" -eq 0 ] && [ "$CACHE_DIRS" -eq 0 ]; then
    echo -e "${GREEN}✅ Nothing to clean! Framework is already in clean state.${NC}"
    exit 0
fi

# Ask about FAISS index
FAISS_EXISTS=false
if [ -f "data/vector_data/merchant_rag_index.faiss" ]; then
    FAISS_EXISTS=true
    echo -e "${YELLOW}⚠️  FAISS index found: data/vector_data/merchant_rag_index.faiss${NC}"
    echo "   This is a generated file but required for the pipeline to work."
    echo "   Do you want to delete it? (y/n)"
    read -p "   > " -n 1 -r
    echo
    DELETE_FAISS=false
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DELETE_FAISS=true
    fi
fi

# Confirmation
echo
echo -e "${YELLOW}⚠️  This will delete all generated files listed above.${NC}"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}ℹ️  Cleanup cancelled.${NC}"
    exit 0
fi

echo
echo "🧹 Cleaning up..."

# 1. Remove output files
if [ "$OUTPUT_FILES" -gt 0 ]; then
    echo -e "${BLUE}   Removing output files...${NC}"
    find output -name "*.csv" -delete 2>/dev/null || true
    find output -name "*.json" -delete 2>/dev/null || true
    echo -e "   ${GREEN}✅ Removed $OUTPUT_FILES output file(s)${NC}"
fi

# 2. Remove test CSV files
if [ "$TEST_FILES" -gt 0 ]; then
    echo -e "${BLUE}   Removing test CSV files...${NC}"
    find data -name "test_*.csv" -delete 2>/dev/null || true
    echo -e "   ${GREEN}✅ Removed $TEST_FILES test file(s)${NC}"
fi

# 3. Remove log files
if [ "$LOG_FILES" -gt 0 ]; then
    echo -e "${BLUE}   Removing log files...${NC}"
    find logs -name "*.log" -delete 2>/dev/null || true
    echo -e "   ${GREEN}✅ Removed $LOG_FILES log file(s)${NC}"
fi

# 4. Remove test scripts
if [ "$TEST_SCRIPTS" -gt 0 ]; then
    echo -e "${BLUE}   Removing test scripts...${NC}"
    find . -maxdepth 1 -name "test_*.py" -delete 2>/dev/null || true
    echo -e "   ${GREEN}✅ Removed $TEST_SCRIPTS test script(s)${NC}"
fi

# 5. Remove cache directories
if [ "$CACHE_DIRS" -gt 0 ]; then
    echo -e "${BLUE}   Removing Python cache directories...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo -e "   ${GREEN}✅ Removed $CACHE_DIRS cache directory(ies)${NC}"
fi

# 6. Remove FAISS index if requested
if [ "$FAISS_EXISTS" = true ] && [ "$DELETE_FAISS" = true ]; then
    echo -e "${BLUE}   Removing FAISS index...${NC}"
    rm -f data/vector_data/merchant_rag_index.faiss 2>/dev/null || true
    rm -f data/vector_data/merchant_rag_metadata.pkl 2>/dev/null || true
    echo -e "   ${GREEN}✅ Removed FAISS index files${NC}"
fi

# 7. Remove any temporary files
echo -e "${BLUE}   Removing temporary files...${NC}"
find . -maxdepth 1 -name "*.tmp" -delete 2>/dev/null || true
find . -maxdepth 1 -name "*.temp" -delete 2>/dev/null || true
echo -e "   ${GREEN}✅ Cleaned temporary files${NC}"

# 8. Keep directory structure
echo -e "${BLUE}   Ensuring directory structure...${NC}"
mkdir -p output
mkdir -p logs
mkdir -p data/vector_data
echo -e "   ${GREEN}✅ Directory structure maintained${NC}"

echo
echo -e "${GREEN}✅ Cleanup completed!${NC}"
echo
echo "📋 Framework Status:"
echo "   ✅ Output directory: Clean"
echo "   ✅ Test files: Removed"
echo "   ✅ Log files: Removed"
echo "   ✅ Cache: Removed"
if [ "$FAISS_EXISTS" = true ] && [ "$DELETE_FAISS" = false ]; then
    echo "   ⚠️  FAISS index: Kept (required for pipeline)"
    echo "      To rebuild: python utils/faiss_index_builder.py"
else
    echo "   ℹ️  FAISS index: Removed"
    echo "      To rebuild: python utils/faiss_index_builder.py"
fi
echo
echo "🚀 Ready for fresh start!"
echo
echo "💡 Next steps:"
echo "   1. Build FAISS index: python utils/faiss_index_builder.py"
echo "   2. Run pipeline: python pipeline.py"
echo "   3. Or use: ./run_e2e_test.sh"

