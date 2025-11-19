# 🚀 IntelliSpend UI Quick Start Guide

## Prerequisites

1. **Python 3.12+** installed
2. **Dependencies** installed (see below)
3. **FAISS Index** built (see below)
4. **LLM API Key** configured (OpenAI or Google Gemini)

---

## 📦 Step 1: Install Dependencies

```bash
# Install required packages
pip install plotly openpyxl

# Or install all dependencies
pip install -r requirements.txt
```

---

## 🔧 Step 2: Build FAISS Index (If Not Already Built)

The FAISS index is required for transaction processing. If you haven't built it yet:

```bash
python utils/faiss_index_builder.py
```

This will create:
- `data/vector_data/merchant_rag_index.faiss`
- `data/vector_data/merchant_rag_metadata.pkl`

**Note:** This only needs to be done once (or when merchant seed is updated).

---

## 🚀 Step 3: Launch the UI

### Option 1: Using Streamlit Command
```bash
streamlit run streamlit_app/app.py
```

### Option 2: Using the Launcher Script
```bash
./run_ui.sh
```

The UI will automatically open in your browser at:
**http://localhost:8501**

---

## 🎯 Step 4: Try It Out!

### 🏠 Dashboard
1. View key metrics and statistics
2. See category distribution charts
3. Review recent transactions
4. Export results (CSV, JSON, Excel)

### 📊 Process Transactions

#### File Upload:
1. Click "📊 Process Transactions" in sidebar
2. Go to "📁 File Upload" tab
3. Upload your CSV file (or use `data/raw_transactions.csv`)
4. Configure column names if needed
5. Click "🚀 Process Transactions"
6. Wait for processing to complete
7. Download results in your preferred format

#### Single Transaction:
1. Go to "✍️ Single Transaction" tab
2. Enter a transaction description (e.g., "STARBUCKS COFFEE #1234")
3. Optionally add amount and date
4. Click "🔍 Classify Transaction"
5. View classification details and explainability

### ✏️ Review & Feedback
1. Click "✏️ Review & Feedback" in sidebar
2. Filter transactions by category, confidence, etc.
3. Review each transaction's classification
4. See similar merchants that were retrieved
5. Submit corrections if needed
6. Export filtered results

### 📈 Analytics
1. View performance metrics
2. See category performance charts
3. Check feedback statistics
4. Analyze trends over time

### ⚙️ Configuration
1. Adjust processing settings (batch size, top K)
2. View LLM configuration
3. Check classification thresholds
4. See current system settings

### 🔄 Apply Feedback
1. View pending feedback count
2. See feedback details
3. Apply feedback to merchant seed
4. Rebuild FAISS index with new patterns

---

## 💡 Example Workflow

### Complete End-to-End Test:

1. **Launch UI**
   ```bash
   streamlit run streamlit_app/app.py
   ```

2. **Process Transactions**
   - Go to "📊 Process Transactions"
   - Upload `data/raw_transactions.csv`
   - Click "🚀 Process Transactions"
   - Wait for completion (~12-15 minutes for 52 transactions)

3. **Review Results**
   - Go to "🏠 Dashboard"
   - Check metrics and charts
   - Review recent transactions

4. **Provide Feedback**
   - Go to "✏️ Review & Feedback"
   - Filter for low confidence transactions
   - Submit corrections for misclassifications
   - Example: Correct "LYFT RIDE" from "Other" to "Transport"

5. **Apply Feedback**
   - Go to "🔄 Apply Feedback"
   - Click "🔄 Apply Feedback & Rebuild Index"
   - System learns from your corrections

6. **Re-process** (Optional)
   - Process transactions again
   - See improved classifications!

---

## 🐛 Troubleshooting

### UI Won't Start
- **Check dependencies**: `pip install plotly openpyxl streamlit`
- **Check Python version**: `python --version` (should be 3.12+)

### "FAISS Index Missing" Error
- Run: `python utils/faiss_index_builder.py`
- Check that `data/vector_data/merchant_rag_index.faiss` exists

### Processing Fails
- Check `.env` file has LLM API keys
- Verify FAISS index is built
- Check logs in `logs/intellispend.log`

### Export Not Working
- For Excel: `pip install openpyxl`
- For JSON/CSV: Should work out of the box

### Slow Processing
- This is normal! LLM calls take time
- Use smaller batch sizes for faster feedback
- Consider using Gemini (free tier) for testing

---

## 📊 Expected Performance

- **Single Transaction**: 10-20 seconds (with LLM)
- **52 Transactions**: ~12-15 minutes (with LLM for all)
- **High Confidence Transactions**: <1 second (direct classification)

---

## 🎨 UI Features Overview

✅ **Modern Design**: Purple gradient theme, professional look
✅ **Interactive Charts**: Plotly-powered visualizations
✅ **Real-time Processing**: Live progress updates
✅ **Export Options**: CSV, JSON, Excel
✅ **Explainability**: See why classifications were made
✅ **Feedback System**: Submit corrections, improve system
✅ **Configuration**: Adjust settings without code changes

---

## 🚀 Quick Commands

```bash
# Install dependencies
pip install plotly openpyxl

# Build FAISS index (if needed)
python utils/faiss_index_builder.py

# Launch UI
streamlit run streamlit_app/app.py

# Or use launcher
./run_ui.sh
```

---

## 💡 Tips

1. **First Time**: Build FAISS index before using UI
2. **Testing**: Use single transaction mode for quick tests
3. **Feedback**: Focus on low-confidence transactions first
4. **Export**: Use filtered results for targeted analysis
5. **Performance**: Smaller batch sizes = faster feedback

---

Enjoy exploring IntelliSpend! 🎉

