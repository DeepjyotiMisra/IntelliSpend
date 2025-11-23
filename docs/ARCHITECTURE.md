# IntelliSpend Architecture & Flow Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [Complete Workflow Example](#complete-workflow-example)

---

## 🎯 System Overview

IntelliSpend is a **multi-agent AI system** that categorizes financial transactions using:
- **RAG (Retrieval-Augmented Generation)**: FAISS vector search for merchant similarity
- **LLM Reasoning**: OpenAI GPT for intelligent classification
- **Agentic Workflow**: Agno framework for orchestration

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    IntelliSpend System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Preprocessor │───▶│  Retriever   │───▶│  Classifier  │   │
│  │    Agent     │    │    Agent     │    │    Agent     │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                    │            │
│         ▼                   ▼                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Data Utils   │    │ FAISS Index  │    │ LLM (OpenAI) │   │
│  │ Normalization│    │   Retriever  │    │  Reasoning   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                    │            │
│         └───────────────────┴────────────────────┘            │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │  Configuration  │                            │
│                  │   & Settings    │                            │
│                  └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Component Architecture

### Layer 1: Configuration Layer
**Purpose**: Centralized settings and taxonomy management

### Layer 2: Utility Layer (Core Components)
**Purpose**: Reusable functions for data processing, embeddings, and retrieval

### Layer 3: Agent Layer
**Purpose**: Intelligent agents that orchestrate the workflow

### Layer 4: Application Layer
**Purpose**: User-facing interfaces and orchestration

---

## 📊 Data Flow

### Complete Transaction Processing Flow

```
1. INPUT: Raw Transaction
   "AMAZON PAY INDIA TXN 12345"
   │
   ▼
2. PREPROCESSOR AGENT
   ├─ Normalizes: "AMAZON PAY INDIA"
   ├─ Extracts: Payment Mode = "UNKNOWN"
   └─ Output: Clean, normalized description
   │
   ▼
3. RETRIEVER AGENT
   ├─ Generates embedding for normalized text
   ├─ Searches FAISS index for similar merchants
   └─ Returns: Top 5 matches with scores
   │
   ▼
4. CLASSIFIER AGENT 
   ├─ Uses retrieved merchants as context
   ├─ LLM reasons about category
   └─ Output: Category + Confidence Score
   │
   ▼
5. OUTPUT: Categorized Transaction
   {
     "merchant": "Amazon",
     "category": "Shopping",
     "confidence": 0.92
   }
```

---

## 📁 File-by-File Breakdown

### 🔧 Configuration Files

#### `config/settings.py`
**Purpose**: Central configuration manager for the entire system

**Key Responsibilities**:
- Loads environment variables from `.env`
- Manages OpenAI/Azure OpenAI configuration
- Stores paths to data files (FAISS index, metadata, seed data)
- Defines thresholds (similarity scores, confidence levels)
- Loads and caches taxonomy configuration
- Provides validation methods

**Key Methods**:
- `Settings.load_taxonomy()`: Loads category taxonomy from JSON
- `Settings.get_categories()`: Returns all categories
- `Settings.validate_config()`: Checks if all required config is present

**Used By**: All other modules import `settings` from here

---

#### `config/taxonomy.json`
**Purpose**: Defines the category taxonomy (user-configurable)

**Structure**:
```json
{
  "categories": [
    {
      "id": "shopping",
      "name": "Shopping",
      "description": "...",
      "keywords": ["amazon", "flipkart", ...]
    },
    ...
  ]
}
```

**Key Features**:
- Easy to modify without code changes
- Supports 11 default categories
- Each category has keywords for matching

**Used By**: `settings.py` loads this, Classifier Agent will use it

---

### 🛠️ Utility Layer (Core Components)

#### `pipeline.py`
**Purpose**: Main orchestration script for processing transactions end-to-end

**Key Functions**:

1. **`process_single_transaction(description, amount, date) -> Dict`**
   - **What it does**: Processes a single transaction through the complete pipeline
   - **Process**:
     ```
     1. Preprocess: Normalize description, extract payment mode
     2. Retrieve: Find similar merchants using FAISS
     3. Classify: Use best match to assign merchant and category
     4. Return: Structured result with merchant, category, confidence
     ```
   - **Returns**: Dict with merchant, category, confidence_score, match_quality, etc.
   - **Used By**: Batch processing, file processing

2. **`process_batch_transactions(descriptions, amounts, dates, batch_size) -> List[Dict]`**
   - **What it does**: Processes multiple transactions in batches
   - **Optimization**: Processes in configurable batch sizes
   - **Used By**: File processing

3. **`process_transactions_file(input_file, output_file, ...) -> pd.DataFrame`**
   - **What it does**: Processes a CSV file of transactions
   - **Process**:
     - Loads transactions from CSV
     - Processes each transaction
     - Generates statistics
     - Saves results to CSV
   - **Returns**: DataFrame with categorized transactions
   - **Used By**: Main entry point, command-line interface

**Current Implementation**:
- **Hybrid Approach**: Uses tools directly for efficiency, Classifier Agent for accuracy
  - **Fast Path**: Direct classification from FAISS matches (high confidence)
  - **LLM Path**: Classifier Agent for low-confidence matches (when enabled)
  - **Result**: Processes 2000 transactions in ~30 seconds (fast mode) or ~10-15 minutes (LLM mode for low-confidence only)
- Preprocessor → Retriever → Smart Classification
  - High confidence (≥threshold): Direct classification (fast)
  - Low confidence (<threshold): Classifier Agent (accurate)
- **Classifier Agent**: ✅ Implemented and integrated
  - Supports OpenAI and Google Gemini
  - Automatically triggered for low-confidence matches when enabled
  - Provides LLM reasoning for ambiguous cases

**Output Columns**:
- `merchant`: Identified merchant name
- `category`: Assigned category
- `confidence_score`: Similarity score (0-1)
- `match_quality`: 'high', 'low', 'none', or 'agent_llm'
- `classification_source`: 'direct', 'llm', or 'direct_fallback'
- `payment_mode`: Payment method extracted
- `num_matches`: Number of similar merchants found
- `processing_status`: Success/error status

**Used By**: Command-line interface, evaluation scripts

---

#### `utils/data_utils.py`
**Purpose**: Data loading, normalization, and preprocessing utilities

**Key Functions**:

1. **`normalize_transaction_string(text: str) -> str`**
   - **What it does**: Cleans and standardizes transaction descriptions
   - **Process**:
     - Converts to uppercase
     - Removes extra whitespace
     - Removes transaction numbers (TXN 12345)
     - Removes special characters
   - **Example**: 
     - Input: `"AMAZON PAY INDIA TXN 12345"`
     - Output: `"AMAZON PAY INDIA"`
   - **Used By**: Preprocessor Agent, FAISS index builder

2. **`load_merchant_seed_data(file_path) -> pd.DataFrame`**
   - **What it does**: Loads merchant seed data from CSV
   - **Process**:
     - Reads CSV file
     - Validates required columns (merchant, description, category)
     - Normalizes all descriptions
   - **Returns**: DataFrame with normalized descriptions
   - **Used By**: FAISS index builder

3. **`load_transactions(file_path) -> pd.DataFrame`**
   - **What it does**: Loads transaction data for processing
   - **Returns**: DataFrame with normalized transaction descriptions
   - **Used By**: Main pipeline, evaluation scripts

4. **`extract_payment_mode(description: str) -> str`**
   - **What it does**: Extracts payment method from description
   - **Returns**: "UPI", "NEFT", "IMPS", "CARD", or "UNKNOWN"
   - **Used By**: Preprocessor Agent

5. **`validate_transaction_data(df) -> (bool, List[str])`**
   - **What it does**: Validates transaction DataFrame quality
   - **Returns**: (is_valid, list_of_errors)
   - **Used By**: Data loading pipelines

---

#### `utils/embedding_service.py`
**Purpose**: Manages sentence-transformers model for generating embeddings

**Key Class**: `EmbeddingService`

**Key Methods**:

1. **`__init__(model_name)`**
   - **What it does**: Initializes the embedding model (lazy loading)
   - **Model**: `sentence-transformers/all-MiniLM-L6-v2` (default)
   - **Process**: Loads model only when first used

2. **`encode(texts, normalize=True) -> np.ndarray`**
   - **What it does**: Converts text to vector embeddings
   - **Input**: String or list of strings
   - **Process**:
     - Encodes text using sentence-transformers
     - Normalizes vectors (L2 normalization) for cosine similarity
     - Converts to float32 for FAISS compatibility
   - **Output**: numpy array of embeddings
   - **Used By**: FAISS index builder, FAISS retriever

3. **`encode_batch(texts, batch_size=128)`**
   - **What it does**: Optimized batch encoding
   - **Used By**: FAISS index builder for processing many merchants

**Singleton Pattern**: `get_embedding_service()` returns global instance

**Used By**: FAISS index builder, FAISS retriever

---

#### `utils/faiss_index_builder.py`
**Purpose**: Builds and manages FAISS vector index for merchant similarity search

**Key Class**: `FAISSIndexBuilder`

**Key Methods**:

1. **`build_index_from_seed_data(seed_file)`**
   - **What it does**: Creates FAISS index from merchant seed data
   - **Process**:
     ```
     1. Load merchant seed CSV
     2. Extract descriptions
     3. Generate embeddings using EmbeddingService
     4. Create FAISS IndexFlatIP (Inner Product = Cosine Similarity)
     5. Add embeddings to index
     6. Create metadata list (merchant, category, etc.)
     ```
   - **Output**: In-memory FAISS index + metadata
   - **Used By**: Initial setup, index rebuilding

2. **`save_index()`**
   - **What it does**: Saves FAISS index and metadata to disk
   - **Saves**:
     - `data/vector_data/merchant_rag_index.faiss` (FAISS index)
     - `data/vector_data/merchant_rag_metadata.pkl` (metadata pickle)
   - **Used By**: After building index

3. **`load_index()`**
   - **What it does**: Loads existing index from disk
   - **Used By**: FAISS retriever initialization

4. **`add_merchants(merchants, descriptions)`**
   - **What it does**: Adds new merchants to existing index (incremental)
   - **Used By**: Feedback Agent (future)

**Convenience Function**: `build_merchant_index()` - one-line index building

**Used By**: Initial setup script, index maintenance

---

#### `utils/generate_merchant_seed.py`
**Purpose**: Automatically generates merchant seed data from raw transactions

**Key Function**: `generate_merchant_seed_from_transactions()`

**What It Does**:
1. Loads raw transactions CSV
2. Extracts merchant names from descriptions using heuristics
3. Categorizes merchants automatically
4. Creates seed data with one example per merchant
5. Saves to `merchants_seed.csv`

**Features**:
- Pattern matching for merchant extraction
- Automatic categorization
- Filters by minimum occurrences
- Creates initial seed data

**Used By**: Initial setup, seed data generation

---

#### `utils/expand_merchant_seed.py`
**Purpose**: Expands merchant seed with multiple transaction patterns per merchant

**Key Function**: `expand_merchant_seed()`

**What It Does**:
1. Loads raw transactions CSV
2. Groups transactions by merchant
3. Extracts multiple transaction patterns per merchant (6-8 patterns)
4. Creates expanded seed data with diverse patterns
5. Saves expanded seed to `merchants_seed.csv`

**Features**:
- Multiple patterns per merchant (TXN, TRANSFER, UPI formats)
- Pattern frequency tracking
- Configurable max patterns per merchant
- Improves matching accuracy significantly

**Impact**:
- Before: 26 records (1 pattern per merchant)
- After: 159 records (6-7 patterns per merchant)
- Result: Higher confidence scores, better matching

**Used By**: Merchant seed expansion, accuracy improvement

---

#### `utils/faiss_retriever.py`
**Purpose**: Performs similarity search on FAISS index to find similar merchants

**Key Class**: `FAISSRetriever`

**Key Methods**:

1. **`__init__(index_path, metadata_path)`**
   - **What it does**: Initializes retriever and loads index
   - **Process**: Automatically loads FAISS index and metadata if they exist
   - **Used By**: Retriever Agent initialization

2. **`retrieve(query, top_k=5, min_score=None) -> List[Dict]`**
   - **What it does**: Finds similar merchants for a query
   - **Process**:
     ```
     1. Normalize query description
     2. Generate embedding using EmbeddingService
     3. Search FAISS index (cosine similarity)
     4. Get top_k results with scores
     5. Filter by min_score if provided
     6. Return formatted results with metadata
     ```
   - **Returns**: List of dicts with:
     - `merchant`: Merchant name
     - `category`: Category name
     - `score`: Similarity score (0-1)
     - `description`: Original description
   - **Used By**: Retriever Agent, tools

3. **`batch_retrieve(queries, top_k=5)`**
   - **What it does**: Batch retrieval for multiple queries
   - **Optimization**: Processes all queries in one FAISS search
   - **Used By**: Batch processing pipelines

4. **`get_best_match(query, min_score=None)`**
   - **What it does**: Returns only the best matching merchant
   - **Used By**: Quick lookups

5. **`explain(query, top_k=5) -> str`**
   - **What it does**: Generates human-readable explanation
   - **Used By**: Debugging, user explanations

**Singleton Pattern**: `get_retriever()` returns global instance

**Used By**: Retriever Agent, tools

---

### 🤖 Agent Layer

#### `agents/tools.py`
**Purpose**: Wrapper functions that agents can call as "tools"

**Key Functions**:

1. **`normalize_transaction(description: str) -> Dict`**
   - **What it does**: Wrapper for `data_utils.normalize_transaction_string()`
   - **Returns**: Structured dict with original, normalized, payment_mode
   - **Used By**: Preprocessor Agent (as a tool)

2. **`retrieve_similar_merchants(description, top_k=5) -> Dict`**
   - **What it does**: Wrapper for `faiss_retriever.retrieve()`
   - **Process**:
     - Normalizes description
     - Calls FAISS retriever
     - Formats results
   - **Returns**: Dict with query, results, best_match
   - **Used By**: Retriever Agent (as a tool)

3. **`batch_normalize_transactions(descriptions: List[str])`**
   - **What it does**: Batch version of normalization
   - **Used By**: Preprocessor Agent for batch processing

4. **`batch_retrieve_merchants(descriptions: List[str])`**
   - **What it does**: Batch version of retrieval
   - **Used By**: Retriever Agent for batch processing

**Why These Exist**: Agno agents need functions (tools) they can call. These wrap our utility functions in a format agents can use.

---

#### `agents/preprocessor_agent.py`
**Purpose**: Agno agent that cleans and normalizes transaction data with LLM reasoning

**Key Function**: `create_preprocessor_agent() -> Agent`

**What It Does**:
1. Creates an Agno Agent instance with Azure OpenAI model
2. Configures it with Azure OpenAI model (LLM-powered)
3. Gives it tools: `normalize_transaction`, `batch_normalize_transactions`
4. Sets agent description and instructions

**Agent Capabilities**:
- Can normalize transaction descriptions
- Can extract payment modes
- Can process batches
- **Uses LLM to understand context and provide explanations**
- **Uses LLM to reason about normalization decisions**

**How It Works**:
```
User/System → Agent.run("Normalize: AMAZON TXN 12345")
           ↓
LLM processes request and understands intent
           ↓
LLM decides to call normalize_transaction tool
           ↓
Tool executes (same as direct call)
           ↓
LLM processes result and provides explanation
           ↓
Returns structured result with LLM-generated explanation
```

**Important Note**: 
- **Pipeline uses tools directly** (bypasses agents) for efficiency
- **Agents are available** for interactive use, explanations, and future enhancements
- **Agents DO use LLM** - they add reasoning and explanation layer on top of tools

**Used By**: Interactive use cases, future Classifier Agent, Streamlit UI (not used by main pipeline for efficiency)

---

#### `agents/retriever_agent.py`
**Purpose**: Agno agent that finds similar merchants using FAISS with LLM reasoning

**Key Function**: `create_retriever_agent() -> Agent`

**What It Does**:
1. Creates an Agno Agent instance with Azure OpenAI model
2. Configures it with Azure OpenAI model (LLM-powered)
3. Gives it tools: `retrieve_similar_merchants`, `batch_retrieve_merchants`
4. Sets agent description and instructions

**Agent Capabilities**:
- Can find similar merchants for transactions
- **Can explain why merchants match (LLM-generated explanations)**
- Can process batches
- **Uses LLM to interpret results and provide context**
- **Uses LLM to reason about similarity scores**

**How It Works**:
```
User/System → Agent.run("Find merchants for: AMAZON PAY")
           ↓
LLM processes request and understands intent
           ↓
LLM decides to call retrieve_similar_merchants tool
           ↓
Tool calls FAISSRetriever.retrieve() (same as direct call)
           ↓
LLM processes results and explains why merchants match
           ↓
Returns results with LLM-generated explanation
```

**Important Note**:
- **Pipeline uses tools directly** (bypasses agents) for efficiency
- **Agents are available** for interactive use, explanations, and debugging
- **Agents DO use LLM** - they add reasoning and explanation layer on top of tools

**Dependencies**: Requires FAISS index to be built first

**Used By**: Interactive use cases, debugging, future enhancements (not used by main pipeline for efficiency)

---

## 🔄 Complete Workflow Example

### Scenario: Categorize transaction "AMAZON PAY INDIA TXN 12345"

#### Step 1: Preprocessing
```
Input: "AMAZON PAY INDIA TXN 12345"
       ↓
Preprocessor Agent called
       ↓
Agent uses normalize_transaction tool
       ↓
Tool calls data_utils.normalize_transaction_string()
       ↓
Output: {
  "original": "AMAZON PAY INDIA TXN 12345",
  "normalized": "AMAZON PAY INDIA",
  "payment_mode": "UNKNOWN"
}
```

#### Step 2: Retrieval
```
Input: "AMAZON PAY INDIA" (normalized)
       ↓
Retriever Agent called
       ↓
Agent uses retrieve_similar_merchants tool
       ↓
Tool calls:
  1. normalize_transaction_string() (re-normalize)
  2. EmbeddingService.encode() (generate embedding)
  3. FAISSRetriever.retrieve() (search index)
       ↓
FAISS Search Process:
  - Query embedding: [0.1, 0.2, ..., 0.5] (384 dimensions)
  - Search index for top 5 matches
  - Return merchants with scores
       ↓
Output: {
  "query": "AMAZON PAY INDIA",
  "results": [
    {
      "merchant": "Amazon",
      "category": "Shopping",
      "score": 0.92,
      "description": "AMAZON PAY INDIA TXN 50852"
    },
    ...
  ],
  "best_match": {...}
}
```

#### Step 3: Classification (Current Implementation)
```
Input: Normalized description + Retrieved merchants
       ↓
Direct Classification (from best match)
       ↓
Process:
  - Get best match from retrieval results
  - Extract merchant and category from best match
  - Calculate confidence from similarity score
  - Determine match_quality (high/low/none)
       ↓
Output: {
  "merchant": "Amazon",
  "category": "Shopping",
  "confidence_score": 0.92,
  "match_quality": "high",
  "payment_mode": "UNKNOWN",
  "num_matches": 3
}
```

**Note**: Classifier Agent (LLM-based reasoning) is now implemented and automatically used for low-confidence matches when enabled via `USE_CLASSIFIER_AGENT=True` or `--use-agent` flag.

---

## 🔗 Component Dependencies

```
config/settings.py
    ↑
    ├── Used by: ALL modules
    └── Loads: taxonomy.json, .env

utils/data_utils.py
    ↑
    ├── Used by: preprocessor_agent, faiss_index_builder
    └── Uses: settings (for paths)

utils/embedding_service.py
    ↑
    ├── Used by: faiss_index_builder, faiss_retriever
    └── Uses: settings (for model name)

utils/faiss_index_builder.py
    ↑
    ├── Uses: data_utils, embedding_service, settings
    └── Creates: FAISS index files

utils/faiss_retriever.py
    ↑
    ├── Uses: embedding_service, settings
    └── Reads: FAISS index files (created by builder)

agents/tools.py
    ↑
    ├── Uses: data_utils, faiss_retriever
    └── Used by: preprocessor_agent, retriever_agent

agents/preprocessor_agent.py
    ↑
    ├── Uses: tools, settings
    └── Provides: Agent interface

agents/retriever_agent.py
    ↑
    ├── Uses: tools, settings
    └── Requires: FAISS index to be built
```

---

## 🎯 Key Design Decisions

### 1. **Separation of Concerns**
- **Utilities**: Pure functions, no agent logic
- **Tools**: Wrappers for agents to use utilities
- **Agents**: Orchestration and LLM reasoning

### 2. **Singleton Patterns**
- `EmbeddingService`: One model instance (expensive to load)
- `FAISSRetriever`: One index instance (large memory)

### 3. **Lazy Loading**
- Models loaded only when needed
- Index loaded on retriever initialization

### 4. **Configuration Management**
- All settings in one place (`settings.py`)
- Environment variables for secrets
- JSON file for taxonomy (user-configurable)

### 5. **Error Handling**
- Graceful fallbacks (Toolkit → direct functions)
- Clear error messages
- Validation at each step

---

## ✅ Current Status

**Implemented**:
- ✅ Preprocessor tools (normalization, payment mode extraction)
- ✅ Retriever tools (FAISS similarity search)
- ✅ Main pipeline (`pipeline.py`) - processes transactions end-to-end
- ✅ Merchant seed generation utilities (comprehensive with multiple patterns)
- ✅ Batch processing with parallel LLM calls
- ✅ Statistics and reporting
- ✅ Classifier Agent for LLM-based reasoning on low-confidence matches
- ✅ Support for OpenAI and Google Gemini models
- ✅ Smart routing: Fast path for high-confidence, LLM path for low-confidence
- ✅ Feedback Agent for continuous learning
- ✅ Streamlit UI with full feature set
- ✅ Custom categories management
- ✅ Cleanup scripts for resetting framework state
- ✅ End-to-end testing guide

**Working Flow**:
```
Raw Transaction → Preprocess → Retrieve → 
  ├─ High Confidence → Direct Classification
  └─ Low Confidence → LLM Classification (Classifier Agent)
  → Output → Feedback Loop (optional)
```

**Streamlit UI Features**:
- ✅ **Dashboard**: Overview, metrics, category distribution, confidence analysis
- ✅ **Process Transactions**: File upload, single transaction classification, progress tracking
- ✅ **Review & Feedback**: Transaction review, corrections, custom category creation
- ✅ **Analytics**: Performance metrics, category trends, feedback statistics
- ✅ **Configuration**: LLM provider selection, processing settings, thresholds
- ✅ **Apply Feedback**: Process pending feedback, update merchant seed, rebuild index
- ✅ **Custom Categories**: Create, manage, and delete custom categories
- ✅ **Merchant Seed Management**: Generate, view, and rebuild merchant seed via UI

**Future Enhancements**:
- Evaluation metrics and benchmarking dashboard
- Advanced analytics and reporting
- Multi-user support and authentication

---

This architecture provides:
- ✅ **Modularity**: Each component has a clear purpose
- ✅ **Reusability**: Utilities can be used independently
- ✅ **Testability**: Each layer can be tested separately
- ✅ **Extensibility**: Easy to add new agents or features
- ✅ **Maintainability**: Clear separation of concerns

