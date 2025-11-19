# IntelliSpend Quick Start Guide

Complete step-by-step guide to set up and run IntelliSpend transaction categorization system.

## 📋 Prerequisites

- Python 3.12+
- LLM API access (OpenAI or Google Gemini)
  - OpenAI: API key or Azure OpenAI endpoint
  - Google Gemini: Free tier available (recommended for testing)
- Virtual environment (conda or venv)

---

## 🚀 Step 1: Environment Setup

### 1.1 Clone and Navigate
```bash
cd /Users/lavi/Documents/git/hackathon_external/IntelliSpend
```

### 1.2 Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n intellispend python=3.12
conda activate intellispend

# OR using venv
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

### 1.3 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔐 Step 2: Configuration

### 2.1 Create Environment File
Create a `.env` file in the project root with your LLM credentials:

```bash
# Option 1: Google Gemini (Recommended - Free tier available)
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL_NAME=gemini-2.0-flash
CLASSIFIER_MODEL_PROVIDER=gemini

# Option 2: OpenAI (Direct)
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4o-mini
CLASSIFIER_MODEL_PROVIDER=openai

# Option 3: Azure OpenAI
OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
MODEL_NAME=gpt-4o-mini  # or your deployment name
MODEL_API_VERSION=2024-02-15-preview
CLASSIFIER_MODEL_PROVIDER=openai

# Optional: Enable Classifier Agent for low-confidence matches
USE_CLASSIFIER_AGENT=False  # Set to True to enable
CLASSIFIER_AGENT_THRESHOLD=0.76  # Confidence threshold to trigger agent
```

**Get API Keys:**
- Google Gemini: https://aistudio.google.com/apikey (Free tier available)
- OpenAI: https://platform.openai.com/api-keys

### 2.2 Verify Configuration Files
Ensure these files exist:
- `config/taxonomy.json` - Category definitions (should already exist)
- `data/raw_transactions.csv` - Input transaction file (should already exist)

---

## 📊 Step 3: Merchant Seed Setup

### 3.1 Generate Comprehensive Merchant Seed

**Option A: Using Streamlit UI (Recommended)**
1. Start the Streamlit UI:
   ```bash
   streamlit run streamlit_app/app.py
   ```
2. Navigate to **"📚 Merchant Seed"** page
3. Upload your transactions CSV file or use existing file path
4. Configure:
   - **Min Occurrences**: Minimum times a merchant must appear (default: 2)
   - **Max Patterns per Merchant**: Maximum transaction patterns per merchant (default: 10)
5. Click **"🔨 Generate Merchant Seed"**
6. Click **"🔨 Rebuild FAISS Index"** to use the new seed

**Option B: Using Command Line**
```bash
python utils/expand_merchant_seed.py
```

This generates a comprehensive merchant seed with multiple transaction patterns per merchant for better matching accuracy.

**Expected Output:**
```
✅ Generated X merchant patterns
✅ Found Y unique merchants
✅ Saved to data/merchants_seed.csv
```

---

## 🔍 Step 4: Build FAISS Index

### 4.1 Build Vector Index
```bash
python utils/faiss_index_builder.py
```

This will:
- Load merchant seed data
- Generate embeddings using sentence-transformers
- Build FAISS vector index
- Save index files:
  - `data/vector_data/merchant_rag_index.faiss` (vector index)
  - `data/vector_data/merchant_rag_metadata.pkl` (merchant metadata)

**Expected Output:**
```
✅ Built FAISS index with X merchants
✅ Index saved to data/vector_data/merchant_rag_index.faiss
✅ Metadata saved to data/vector_data/merchant_rag_metadata.pkl
```

---

## 🎯 Step 5: Process Transactions

### 5.1 Using Streamlit UI (Recommended)
```bash
streamlit run streamlit_app/app.py
```

Navigate to **"📊 Process Transactions"** page:
- Upload CSV file or enter single transaction
- Configure column names if needed
- View real-time progress
- Export results (CSV, JSON, Excel)

### 5.2 Using Command Line
```bash
python pipeline.py
```

This processes `data/raw_transactions.csv` and saves results to `output/categorized_transactions.csv`.

### 5.3 Custom Input/Output
```bash
python pipeline.py --input data/my_transactions.csv --output output/my_results.csv
```

### 5.4 Custom Column Names
```bash
python pipeline.py --description-col "transaction_desc" --amount-col "value" --date-col "txn_date"
```

### 5.5 Custom Batch Size
```bash
python pipeline.py --batch-size 200
```

### 5.6 Use Classifier Agent (LLM Mode)
Enable Classifier Agent for low-confidence matches (slower but more accurate):

```bash
python pipeline.py --use-agent
```

This will use LLM reasoning for transactions with confidence below the threshold (default: 0.76).

**Expected Output:**
```
🚀 IntelliSpend Transaction Processing Pipeline
================================================================================

Loading transactions from data/raw_transactions.csv...
Loaded 2001 transactions
Processing 2001 transactions in batches of 100...
Processing batches: 100%|████████████| 21/21 [00:45<00:00,  2.15s/it]

✅ Processed 2001 transactions in 45.23 seconds
✅ Average processing time: 22.59 ms per transaction
✅ Results saved to output/categorized_transactions.csv

📊 Processing Statistics:
   Total transactions: 2001
   Successful: 1985 (99.2%)
   Errors: 16 (0.8%)
```

---

## 🧪 Step 6: Test Agents (Optional)

### 6.1 Test Preprocessor Agent
```bash
python agents/preprocessor_agent.py
```

### 6.2 Test Retriever Agent
```bash
python agents/retriever_agent.py
```

### 6.3 Test Classifier Agent
```bash
python agents/classifier_agent.py
```

**Note:** These agents use LLM for reasoning and explanations. They're slower but provide detailed explanations.

## 🧹 Step 7: Cleanup (Optional)

Reset the framework to clean state:

```bash
# Bash version
./cleanup.sh

# OR Python version
python cleanup.py
```

This removes:
- Output files
- Test files
- Log files
- Python cache
- Optionally FAISS index (you'll be asked)

See [CLEANUP_README.md](../CLEANUP_README.md) for details.

---

## 📁 Output Files

After processing, you'll find:

- `output/categorized_transactions.csv` - Categorized transactions with:
- `merchant`: Identified merchant name
- `category`: Assigned category
- `confidence_score`: Similarity score (0-1)
- `payment_mode`: Payment method (UPI, NEFT, IMPS, CARD, etc.)
- `match_quality`: high/low/none/agent_llm
- `classification_source`: direct/llm/direct_fallback
- `num_matches`: Number of similar merchants found
- `top_matches`: Top 3 matching merchants

---

## 🔄 Complete Setup Workflow (One-Time)

For first-time setup, run these commands in order:

```bash
# 1. Activate environment
conda activate intellispend  # or: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file (manually with your credentials)

# 4. Generate comprehensive merchant seed (creates multiple patterns per merchant)
python utils/expand_merchant_seed.py

# 5. Build FAISS index
python utils/faiss_index_builder.py

# 6. Process transactions
python pipeline.py

# OR use the Streamlit UI for all of the above:
streamlit run streamlit_app/app.py
```

---

## 🔄 Daily Usage Workflow

Once setup is complete, you only need:

```bash
# 1. Activate environment
conda activate intellispend

# 2. Process new transactions
python pipeline.py --input data/new_transactions.csv --output output/new_results.csv
```

**Note:** You only need to rebuild the FAISS index if you update `merchants_seed.csv`.

---

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Solution: Ensure virtual environment is activated
conda activate intellispend
# OR
source venv/bin/activate
```

### Issue: FAISS index not found
```bash
# Solution: Build the index first
python utils/faiss_index_builder.py
```

### Issue: Missing .env file
```bash
# Solution: Create .env file with your OpenAI/Azure credentials
# See Step 2.1 above
```

### Issue: Low confidence scores
```bash
# Solution: Generate comprehensive merchant seed for better matching
python utils/expand_merchant_seed.py
python utils/faiss_index_builder.py  # Rebuild index

# OR use Streamlit UI: Navigate to "📚 Merchant Seed" page
```

### Issue: Processing is slow
```bash
# Solution: Use batch processing (already enabled)
# For 100K+ transactions, consider optimizing to use batch_retrieve_merchants()
```

---

## 📊 Verify Setup

Check that all required files exist:

```bash
# Check data files
ls -la data/merchants_seed.csv
ls -la data/vector_data/merchant_rag_index.faiss
ls -la data/vector_data/merchant_rag_metadata.pkl
ls -la data/raw_transactions.csv

# Check config
ls -la config/taxonomy.json
ls -la .env

# Check output directory
ls -la output/
```

---

## 🎯 Next Steps

After successful setup:

1. **Review Results**: Check `output/categorized_transactions.csv`
2. **Filter Low Confidence**: Review transactions with `match_quality='low'`
3. **Evaluate Performance**: Calculate F1-scores and confusion matrix
4. **Improve Seed**: Add more merchants to `merchants_seed.csv` if needed
5. **Rebuild Index**: Run `python utils/faiss_index_builder.py` after seed updates

---

## 📝 Quick Reference

| Task | Command |
|------|---------|
| Start Streamlit UI | `streamlit run streamlit_app/app.py` |
| Generate merchant seed | `python utils/expand_merchant_seed.py` |
| Build FAISS index | `python utils/faiss_index_builder.py` |
| Process transactions | `python pipeline.py` |
| Test preprocessor agent | `python agents/preprocessor_agent.py` |
| Test retriever agent | `python agents/retriever_agent.py` |
| Test classifier agent | `python agents/classifier_agent.py` |
| Cleanup framework | `./cleanup.sh` or `python cleanup.py` |

---

## 💡 Tips

- **First Run**: Generate comprehensive merchant seed before building index for better accuracy
- **UI vs CLI**: Use Streamlit UI for interactive workflows; use CLI for batch/automated processing
- **Performance**: Current pipeline processes ~2000 transactions in ~30-45 seconds
- **Batch Size**: Default is 100; increase for larger datasets (100K+)
- **Agents vs Tools**: Pipeline uses tools directly (fast); agents are for explanations (slower)
- **Index Updates**: Rebuild FAISS index only when `merchants_seed.csv` changes

---

## 📞 Support

For issues or questions:
1. Check `logs/intellispend.log` for detailed error messages
2. Verify all prerequisites are met
3. Ensure `.env` file is correctly configured
4. Check that FAISS index is built before processing

---

## 📚 Additional Resources

- **[E2E_TEST_GUIDE.md](../E2E_TEST_GUIDE.md)** - End-to-end testing guide
- **[CLEANUP_README.md](../CLEANUP_README.md)** - Cleanup script documentation
- **[docs/RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md)** - RAG implementation details
- **[docs/RAG_CONCEPTS_EXPLAINED.md](RAG_CONCEPTS_EXPLAINED.md)** - RAG concepts explained

---

**Last Updated**: 2025-01-05

