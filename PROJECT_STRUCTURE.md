# IntelliSpend - Clean Project Structure

## 📁 Project Organization

```
IntelliSpend/
├── 📄 README.md                    # Project documentation
├── 📄 main.py                      # Main entry point
├── 📄 requirements.txt             # Dependencies
├── 📄 sample_transactions.csv      # Sample data
├── 📄 .env.example                 # Environment template
├── 🔧 .gitignore                   # Git ignore rules
│
├── 🤖 agents/                      # AI Agent System
│   ├── agent_team.py              # Multi-agent orchestrator
│   ├── classifier_agent.py        # Transaction classifier
│   ├── embedding_agent.py         # Text embeddings
│   ├── preprocessor_agent.py      # Data preprocessing
│   ├── retriever_agent.py         # Knowledge retrieval
│   ├── feedback_agent.py          # Learning system
│   ├── classifier.py              # Legacy classifier
│   └── coordinator.py             # Agent coordination
│
├── 🌐 web/                         # Streamlit Web Interface
│   └── app.py                      # Main web application
│
├── ⚙️ config/                      # Configuration
│   └── config.py                   # Settings and API keys
│
├── 📊 models/                      # Data Models
│   └── transaction.py             # Transaction data structure
│
├── 🔧 utils/                       # Utilities
│   ├── data_processing.py         # Data handling utilities
│   └── vector_store.py            # Vector database
│
├── 🧪 tests/                       # Test Suite
│   └── test_suite.py              # Comprehensive tests
│
└── 💾 data/                        # Data Storage
    └── embeddings_cache.pkl       # Cached embeddings
```

## 🧹 Cleanup Summary

### ✅ Removed Files:
- Development test files (`test_ui_*.py`, `test_complex_transactions.py`)
- Temporary documentation (`ENHANCEMENT_*.md`, `UI_FIXES_*.md`)
- Setup and generation scripts (`setup_check.py`, `generate_sample_data.py`)
- Log files (`intellispend.log`)
- Large sample data (`sample_transactions_large.json`)
- IDE configurations (`.idea/`, `.vscode/`)
- Virtual environments (`.venv/`, `venv/`)
- Python cache (`__pycache__/` directories)
- Old agent backups (`agent_team_old.py`, `coordinator_old_backup.py`)
- Empty directories (`data/feedback/`, `data/vectors/`)

### ✅ Kept Essential Files:
- Core application files (`main.py`, `web/app.py`)
- Agent system (all current agent files)
- Configuration and utilities
- Main documentation (`README.md`)
- Test suite (`tests/test_suite.py`)
- Sample data (`sample_transactions.csv`)
- Dependencies (`requirements.txt`)

## 🚀 Usage

### Web Interface:
```bash
streamlit run web/app.py
```

### Command Line:
```bash
python main.py --web                # Launch web interface
python main.py --demo               # Run demo
python main.py --test               # Run tests
python main.py --file <path>        # Process file
```

## 📋 Clean Development

The project is now cleaned of:
- ❌ Temporary test files
- ❌ Development documentation drafts
- ❌ IDE-specific configurations
- ❌ Cached Python bytecode
- ❌ Duplicate/backup files
- ❌ Large unnecessary data files

The `.gitignore` has been updated to prevent these files from being committed in the future.