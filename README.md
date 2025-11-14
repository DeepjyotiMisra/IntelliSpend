# IntelliSpend – Automated AI-Based Financial Transaction Categorisation

![IntelliSpend Banner](https://img.shields.io/badge/IntelliSpend-AI--Powered%20Transaction%20Categorisation-blue?style=for-the-badge)

[![Python](https://img.shields.io/badge/Python-3.12+-brightgreen)](https://python.org)
[![Agno](https://img.shields.io/badge/Agno-2.2+-orange)](https://agno.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-red)](https://github.com/yourusername/intellispend)

## 🎯 Problem Overview

Financial institutions and fintech applications rely heavily on accurate transaction categorisation to deliver insights, budgeting tools, fraud analysis, and financial recommendations. Today, most organisations depend on **expensive third-party APIs**, which introduce:

- ⚡ **Latency** and performance bottlenecks
- 🔒 **Limited flexibility** and customization
- 💰 **Substantial operational costs**
- 🔍 **Poor explainability** (black-box systems)
- 🌍 **Poor adaptation** to regional merchant variations

There is a pressing need for a **cost-efficient, transparent, and high-accuracy in-house AI solution** that can automatically classify transactions like "Uber," "Amazon.com," "HP Petrol Pump" into intuitive categories like Transport, Shopping, Fuel, etc.

## 🚀 Proposed Solution

**IntelliSpend** is an end-to-end, autonomous transaction categorisation system that eliminates the need for costly external APIs. It combines open-source AI components (FAISS, sentence-transformers) with a reasoning-powered LLM and an agentic workflow to deliver exceptional accuracy, transparency, and **full data ownership**.

Unlike solutions like Plaid or Yodlee, IntelliSpend operates entirely within your infrastructure—ensuring **privacy, cost control, and customisation**.

## 💡 Innovation

### 1. Hybrid Retrieval + Reasoning Pipeline (RAG + Agents)

IntelliSpend uses a **multi-agent architecture**:

- **🔧 Preprocessor Agent** — Cleans, normalises, and enriches transaction data
- **🔍 Retriever Agent** — Converts merchant/category knowledge into vector embeddings using sentence-transformers and stores them in a FAISS index. For every incoming transaction, it retrieves the most semantically similar merchants
- **🎯 Classifier Agent** — Uses retrieved candidates, payment metadata, and contextual cues to assign an accurate category along with a confidence score
- **📈 Feedback Agent** — Collects user corrections and updates embeddings dynamically, enabling continuous improvement

This RAG-style pipeline dramatically reduces hallucinations, boosts accuracy for unseen merchants, and provides **interpretable results**.

### 2. Zero-Training Deployment

IntelliSpend works entirely on precomputed embeddings and LLM reasoning. This eliminates the need for model retraining, making the system:

✅ **Lightweight**  
✅ **Flexible**  
✅ **Cost-efficient**  
✅ **Easy to onboard** for banks and fintech startups

### 3. Continuous Feedback & Learning

User feedback on low-confidence predictions is immediately integrated into the knowledge base. This gives IntelliSpend the ability to learn from real-world behaviour without heavy MLOps or retraining pipelines.

Over time, the system becomes:
- 📊 **More accurate**
- 🌍 **More localised**
- 👤 **More personalised**

## 🛠 Technical Approach

| Component | Technology |
|-----------|------------|
| **Languages & Frameworks** | Python, Agno, Streamlit |
| **Vector Store** | FAISS for fast merchant similarity search |
| **LLM Integration** | OpenAI GPT for contextual classification and reasoning |
| **Agent Layer** | Agno for orchestrating multi-agent workflows |
| **Tooling** | MCP server for secure system interaction |
| **Explainability** | Transparent logs with optional SHAP/LIME visualisation |
| **Frontend** | Streamlit demo with live categorisation, confidence scores, and instant feedback updates |

## 📊 Evaluation & Metrics

The evaluation pipeline targets:

- 🎯 **Macro F1-score ≥ 0.90**
- 📈 **Confusion matrix & per-category performance**
- ⚡ **Latency benchmarking** (FAISS retrieval < 50 ms)
- 🚀 **Throughput scaling** for production environments

## 🚦 Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API access or Azure OpenAI endpoint
- Conda (recommended) or virtual environment

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/intellispend.git
cd intellispend
```

2. **Create and activate conda environment**
```bash
conda create -n intellispend python=3.12
conda activate intellispend
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy `.env.example` to `.env` and fill in your configuration:

```bash
cp .env.example .env
```

Edit `.env` file:
```env
# Azure OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
MODEL_API_VERSION=2024-08-01-preview
MODEL_NAME=gpt-4o
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint.com
```

5. **Run the basic agent**
```bash
cd InitialAgents
python basicAgents.py
```

### Example Usage

```python
from InitialAgents.basicAgents import create_intellispend_agent

# Create the agent
agent = create_intellispend_agent()

# Ask financial questions
agent.print_response("What are some effective budgeting strategies?")
agent.print_response("How should I categorize my expenses?")
```

## 📁 Project Structure

```
IntelliSpend/
├── InitialAgents/
│   └── basicAgents.py          # Basic agent implementation
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── .env                       # Your configuration (git-ignored)
└── README.md                  # This file
```

## 🎯 Current Status

🔄 **In Development** - This is the initial implementation focusing on:

- ✅ **Agent Framework Setup** (Agno)
- ✅ **Azure OpenAI Integration**
- ✅ **Basic Financial Q&A Capabilities**
- ✅ **Web Search Integration** (DuckDuckGo)

## 🛣 Roadmap

### Phase 1: Foundation (Current)
- [x] Basic agent setup with Agno
- [x] Azure OpenAI integration
- [x] Environment configuration
- [ ] Transaction preprocessing pipeline

### Phase 2: Core Functionality
- [ ] FAISS vector store implementation
- [ ] Merchant similarity search
- [ ] Transaction classification logic
- [ ] Confidence scoring system

### Phase 3: Advanced Features
- [ ] Multi-agent architecture
- [ ] Continuous learning pipeline
- [ ] Streamlit web interface
- [ ] Performance optimization

### Phase 4: Production Ready
- [ ] Comprehensive evaluation
- [ ] API endpoints
- [ ] Docker containerization
- [ ] Documentation & tutorials

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 💼 Impact

- **👥 Consumers**: Clearer spending insights and improved budgeting tools
- **🏦 Banks & PFMs**: In-house categorisation = lower cost, higher privacy, better explainability
- **🚀 Businesses**: Minimal maintenance and flexible customisation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Agno](https://agno.ai) for the agent framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [sentence-transformers](https://www.sbert.net/) for embedding models
- [Streamlit](https://streamlit.io/) for rapid prototyping

## 📞 Contact

For questions or collaboration opportunities, please open an issue or reach out to the development team.

---

**IntelliSpend** combines Hybrid RAG retrieval, LLM reasoning, and agent-driven automation to deliver a next-generation, fully explainable, and highly scalable transaction categorisation system. Its zero-training adaptability and cost efficiency make it a practical and powerful alternative to third-party categorisation APIs.

⭐ **Star this repo** if you find IntelliSpend useful!