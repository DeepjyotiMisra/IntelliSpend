# IntelliSpend Streamlit UI

High-end, modern web interface for IntelliSpend transaction categorization system.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install plotly>=5.17.0
```

### 2. Launch the UI

```bash
# Option 1: Using the launcher script
./run_ui.sh

# Option 2: Direct command
streamlit run streamlit_app/app.py
```

The UI will open in your browser at `http://localhost:8501`

## 📱 Features

### 🏠 Dashboard
- Key metrics and statistics
- Category distribution charts
- Confidence score visualization
- Recent transactions table

### 📊 Process Transactions
- **File Upload**: Upload CSV files and process in batch
- **Single Transaction**: Test individual transactions in real-time
- Progress tracking and status updates

### ✏️ Review & Feedback
- Review all categorized transactions
- Filter by category, confidence, etc.
- Submit corrections for misclassifications
- Feedback is stored and can be applied to improve the system

### 📈 Analytics
- Performance metrics
- Category performance analysis
- Feedback statistics
- Visual charts and graphs

### 🔄 Apply Feedback
- View pending feedback
- Apply feedback to merchant seed
- Rebuild FAISS index with new patterns
- Improve future classifications

## 🎨 Design Features

- **Modern Gradient Design**: Beautiful purple gradient theme
- **Interactive Charts**: Plotly-powered visualizations
- **Responsive Layout**: Works on all screen sizes
- **Real-time Updates**: Live processing and feedback
- **User-friendly**: Intuitive navigation and clear feedback

## 📋 Requirements

- Python 3.12+
- Streamlit >= 1.28.0
- Plotly >= 5.17.0
- FAISS index built (run `python utils/faiss_index_builder.py`)

## 💡 Usage Tips

1. **First Time**: Build FAISS index before using the UI
2. **Processing**: Use file upload for batch processing
3. **Feedback**: Review low-confidence transactions and provide corrections
4. **Improvement**: Apply feedback regularly to improve accuracy
5. **Analytics**: Monitor performance metrics to track improvements

