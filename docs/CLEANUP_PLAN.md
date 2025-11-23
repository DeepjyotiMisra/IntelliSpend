# Cleanup Plan - Files to Keep vs Remove

## ✅ KEEP (Core Production Code)

### Source Code
- `pipeline.py` - Main pipeline
- `config/` - All config files (settings.py, taxonomy.json, __init__.py)
- `utils/` - All utility files (core functionality)
- `agents/` - Core agent files:
  - `preprocessor_agent.py`
  - `retriever_agent.py`
  - `tools.py`
  - `__init__.py`

### Documentation
- `README.md` - Main documentation
- `ARCHITECTURE.md` - Architecture documentation
- `FLOW_DIAGRAM.md` - Flow diagrams
- `README_PIPELINE.md` - Pipeline usage

### Configuration
- `requirements.txt`
- `.gitignore`
- `data/raw_transactions.csv` - Input example (tracked)
- `data/merchants_seed.csv.example` - Template

---

## ❌ REMOVE (Test/Demo/Temporary)

### Test Files
- `test_pipeline.py`
- `test_expansion_improvement.py`
- `setup_check.py`

### Demo Files
- `agents/demo_preprocessor_retriever.py`
- `InitialAgents/` - Early prototype (not used)

### Temporary Documentation
- `INITIALAGENTS_ANALYSIS.md`
- `TEST_RESULTS.md`
- `TESTING_SUMMARY.md`
- `MERCHANT_SEED_EXPANSION.md`

### Generated Files (already in .gitignore)
- `data/merchants_seed.csv` - Can be regenerated
- `data/*.faiss` - Generated index
- `data/*.pkl` - Generated metadata
- `output/*.csv` - Generated outputs
- `logs/*.log` - Log files
- `__pycache__/` - Python cache
- `path/` - Virtual environment

