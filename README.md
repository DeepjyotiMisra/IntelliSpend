# 🧠 IntelliSpend - AI-Powered Financial Transaction Categorization

![IntelliSpend Banner](https://img.shields.io/badge/IntelliSpend-AI--Powered%20Transaction%20Categorisation-blue?style=for-the-badge)

[![Python](https://img.shields.io/badge/Python-3.12+-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)](https://github.com/DeepjyotiMisra/IntelliSpend)

## 🌟 Overview

**IntelliSpend** is an end-to-end, autonomous transaction categorization system that eliminates the need for costly third-party APIs. It combines open-source AI components (FAISS, sentence-transformers) with reasoning-powered LLMs and an agentic workflow to deliver exceptional accuracy, transparency, and **full data ownership**.

### 💡 Key Benefits

- **🚀 Cost-Effective**: Eliminate expensive API fees and reduce operational costs by up to 80%
- **🎯 High Accuracy**: Multi-strategy AI achieves 90%+ categorization accuracy
- **🔍 RAG-Powered**: Vector similarity search for precise merchant matching
- **🤖 Multi-Agent System**: Intelligent agents for preprocessing, retrieval, classification, and feedback
- **🔄 Continuous Learning**: User feedback automatically improves the system
- **🔒 Secure**: Process sensitive financial data within your own infrastructure
- **⚡ Fast**: Process 2,000+ transactions in minutes with real-time progress tracking
- **🌐 User-Friendly**: Beautiful Streamlit web interface for all operations

## 🏗️ Architecture

IntelliSpend uses a **hybrid RAG (Retrieval-Augmented Generation) + Multi-Agent Architecture**:

- **🔧 Preprocessor Agent** — Cleans, normalises, and enriches transaction data
- **🔍 Retriever Agent** — Converts merchant/category knowledge into vector embeddings and retrieves similar merchants using FAISS
- **🎯 Classifier Agent** — Uses retrieved candidates and LLM reasoning to assign accurate categories with confidence scores
- **📈 Feedback Agent** — Collects user corrections and updates the knowledge base dynamically

**Smart Routing**: High-confidence matches use direct classification (fast), while low-confidence matches use LLM reasoning (accurate).

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key or Google Gemini API key
- 4GB+ RAM (for embedding model and FAISS index)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DeepjyotiMisra/IntelliSpend.git
cd IntelliSpend
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:
```env
# OpenAI Configuration (choose one)
OPENAI_API_KEY=your-openai-api-key
# OR
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Google Gemini Configuration (optional)
GOOGLE_API_KEY=your-google-api-key

# Model Selection (default: gemini)
CLASSIFIER_MODEL_PROVIDER=openai  # or 'gemini'
FEEDBACK_MODEL_PROVIDER=openai    # or 'gemini'
```

5. **Generate Merchant Seed** (First-time setup)

The merchant seed is required for transaction processing. You can generate it via the UI (see below) or via command line:

```bash
python utils/expand_merchant_seed.py --input data/raw_transactions.csv --output data/merchants_seed.csv
python utils/faiss_index_builder.py
```

## 🖥️ Using the Web Interface

**IntelliSpend is designed to be used entirely through the Streamlit web interface.**

### Launch the UI

```bash
streamlit run streamlit_app/app.py
```

The interface will open at `http://localhost:8501`

### Main Features

#### 📊 Dashboard
- Overview of processed transactions
- Category distribution charts
- Confidence score analysis
- Recent transactions view

#### 📊 Process Transactions
- **File Upload**: Upload CSV files with transactions
- **Single Transaction**: Classify individual transactions
- **Auto-detection**: Automatically detects date, description, and amount columns
- **Real-time Progress**: Live progress tracking with ETA
- **Export Options**: Download results as CSV, JSON, or Excel

#### ✏️ Review & Feedback
- Review categorized transactions
- Correct misclassifications
- Create custom categories
- Submit feedback for continuous learning

#### 📈 Analytics
- Processing statistics
- Category trends
- Confidence distribution
- LLM vs Direct classification metrics

#### ⚙️ Configuration
- Select LLM provider (OpenAI or Gemini)
- Adjust processing thresholds
- Configure batch sizes
- View system status

#### 🔄 Apply Feedback
- View pending feedback
- Apply corrections to merchant seed
- Rebuild FAISS index with new data
- Delete unwanted feedback

#### 🏷️ Custom Categories
- Create custom transaction categories
- Add descriptions for categories
- Delete custom categories
- View category usage statistics

#### 📚 Merchant Seed Management
- View current merchant seed
- Generate comprehensive merchant seed from transactions
- Rebuild FAISS index
- Upload transaction files for seed generation

## 📋 Input Format

Your transaction CSV file should have these columns (column names are auto-detected):

- **date**: Transaction date (e.g., "2024-01-15")
- **description**: Transaction description (e.g., "AMAZON PAY INDIA TXN 12345")
- **amount**: Transaction amount (e.g., -1500.00)

**Example**:
```csv
date,description,amount
2024-01-15,"AMAZON PAY INDIA TXN 12345",-1500.00
2024-01-16,"UBER TRIP MUMBAI",-250.00
2024-01-17,"STARBUCKS COFFEE",-150.00
```

## 📊 Output Format

The system generates a categorized CSV with:

- **Original columns**: date, description, amount
- **merchant**: Identified merchant name
- **category**: Assigned category (Shopping, Food & Dining, Transport, etc.)
- **confidence_score**: Confidence level (0-1)
- **classification_source**: 'direct' or 'llm'
- **payment_mode**: Payment method extracted
- **num_matches**: Number of similar merchants found

## 🎯 Default Categories

- Shopping
- Food & Dining
- Transport
- Bills & Payments
- Entertainment
- Grocery
- Healthcare
- Education
- Travel
- Banking
- Other

**Custom categories** can be created through the UI.

## ⚡ Performance

- **Processing Speed**: 2,000 transactions in ~30 seconds (high confidence) or ~10-15 minutes (with LLM)
- **Accuracy**: 90%+ categorization accuracy
- **Confidence Distribution**: 70% high confidence (direct), 30% low confidence (LLM)
- **FAISS Search**: Sub-millisecond similarity search
- **Batch Processing**: Optimized batch embedding generation and parallel LLM calls

## 🔒 Security & Privacy

- **Local Processing**: All data processed within your infrastructure
- **API Keys**: Stored securely in `.env` file (never committed)
- **No Data Sharing**: Transaction data only sent to LLM APIs for classification (no sensitive financial data)
- **User Control**: Full control over data retention and deletion

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.12+ |
| **Agent Framework** | Agno 2.2+ |
| **Vector Store** | FAISS |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| **LLM** | OpenAI GPT-4o / Google Gemini 2.0 Flash |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |

## 📚 Documentation

For detailed documentation, see the `docs/` folder:

- **[QUICKSTART.md](docs/QUICKSTART.md)**: Detailed setup and usage guide
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System architecture and design
- **[RAG_IMPLEMENTATION.md](docs/RAG_IMPLEMENTATION.md)**: RAG pipeline details

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and code of conduct.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Agno](https://agno.ai) for the agent framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [sentence-transformers](https://www.sbert.net/) for text embeddings
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenAI](https://openai.com/) and [Google](https://ai.google.dev/) for LLM APIs

---

**IntelliSpend** combines Hybrid RAG retrieval, LLM reasoning, and agent-driven automation to deliver a next-generation, fully explainable, and highly scalable transaction categorization system.

⭐ **Star this repo** if you find IntelliSpend useful!
