# When to Use Pipeline.py vs LLM Agents

## 🎯 Quick Decision Guide

```
┌─────────────────────────────────────────────────────────────┐
│  Processing 100+ transactions?                              │
│  → Use pipeline.py (fast, free)                             │
├─────────────────────────────────────────────────────────────┤
│  Need explanations for a few transactions?                  │
│  → Use LLM Agents (reasoning, explanations)                 │
├─────────────────────────────────────────────────────────────┤
│  Production batch processing?                               │
│  → Use pipeline.py                                           │
├─────────────────────────────────────────────────────────────┤
│  Debugging or understanding why a transaction was           │
│  categorized a certain way?                                  │
│  → Use LLM Agents                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Comparison Table

| Aspect | pipeline.py (Tools) | LLM Agents |
|--------|---------------------|------------|
| **Speed** | ~15ms per transaction | ~2-5 seconds per transaction |
| **Cost** | FREE (no API calls) | ~$0.001-0.01 per transaction |
| **Throughput** | 1000+ transactions/min | 12-30 transactions/min |
| **Accuracy** | High (deterministic) | High (with reasoning) |
| **Explainability** | Basic (scores, matches) | Rich (LLM explanations) |
| **Best For** | Batch processing, production | Interactive, debugging, edge cases |
| **Scalability** | Excellent (100K+ transactions) | Limited (cost/speed) |

---

## 🚀 Use Pipeline.py When:

### 1. **Batch Processing Large Datasets**
```bash
# Process 10,000 transactions
python pipeline.py --input data/large_batch.csv
```
- ✅ Fast: ~15ms per transaction
- ✅ Free: No LLM API costs
- ✅ Scalable: Handles 100K+ transactions efficiently

**Example Use Cases:**
- Monthly transaction categorization
- Bulk import from bank statements
- Production data processing
- Scheduled batch jobs

### 2. **Production Systems**
```python
# In your production code
from pipeline import process_transactions_file

results = process_transactions_file(
    input_file='data/transactions.csv',
    output_file='output/categorized.csv'
)
```
- ✅ Reliable: Deterministic results
- ✅ Fast: Low latency
- ✅ Cost-effective: No per-transaction costs

### 3. **High-Confidence Transactions**
When transactions match well with merchant seed:
- ✅ Confidence score ≥ 0.76 (high match)
- ✅ Clear merchant patterns
- ✅ Standard transaction formats

### 4. **Automated Workflows**
- Scheduled jobs
- API endpoints
- Background processing
- ETL pipelines

---

## 🤖 Use LLM Agents When:

### 1. **Interactive Exploration & Debugging**
```python
from agents.preprocessor_agent import create_preprocessor_agent

agent = create_preprocessor_agent()
response = agent.run("Normalize: AMAZON PAY INDIA TXN 12345")
# Gets LLM explanation of normalization process
```
- ✅ Understand WHY a transaction was normalized
- ✅ See LLM reasoning process
- ✅ Get detailed explanations

**Example Use Cases:**
- Understanding edge cases
- Debugging categorization issues
- Learning how the system works
- Interactive demos

### 2. **Low-Confidence Transactions**
When pipeline returns low confidence:
```python
# Pipeline returns low confidence
result = process_single_transaction("UNKNOWN MERCHANT XYZ")
if result['match_quality'] == 'low':
    # Use agent to reason about it
    agent = create_retriever_agent()
    explanation = agent.run(f"Explain why this transaction is hard to categorize: {result['original_description']}")
```
- ✅ LLM can reason about ambiguous cases
- ✅ Provides context-aware explanations
- ✅ Can suggest improvements

### 3. **User-Facing Explanations**
When users ask "Why was this categorized as Shopping?"
```python
from agents.retriever_agent import create_retriever_agent

agent = create_retriever_agent()
response = agent.run(
    "Explain why 'AMAZON PAY INDIA' was categorized as Shopping. "
    "Show the reasoning and similar merchants found."
)
# Returns: Rich explanation with reasoning
```
- ✅ User-friendly explanations
- ✅ Context-aware responses
- ✅ Educational value

### 4. **Edge Cases & Ambiguous Transactions**
```python
# Unusual transaction format
agent = create_preprocessor_agent()
response = agent.run(
    "This transaction is ambiguous: 'PAYMENT TO MERCHANT 12345'. "
    "Can you analyze it and suggest normalization?"
)
```
- ✅ LLM reasoning for complex cases
- ✅ Handles ambiguity better
- ✅ Provides suggestions

### 5. **Development & Testing**
- Testing agent behavior
- Validating LLM reasoning
- Understanding system behavior
- Creating documentation examples

---

## 🔄 Hybrid Approach (Recommended)

### Best Practice: Use Both

```python
# 1. Process bulk with pipeline (fast, free)
results = process_transactions_file('data/transactions.csv')

# 2. Filter low-confidence results
low_confidence = results[results['match_quality'] == 'low']

# 3. Use agents for low-confidence cases (reasoning, explanations)
if len(low_confidence) > 0:
    agent = create_retriever_agent()
    for idx, row in low_confidence.iterrows():
        explanation = agent.run(
            f"Analyze this low-confidence transaction: {row['original_description']}"
        )
        # Store explanation for review
```

**Benefits:**
- ✅ Fast processing for bulk (pipeline)
- ✅ Rich explanations for edge cases (agents)
- ✅ Cost-effective (only use LLM when needed)
- ✅ Best of both worlds

---

## 💰 Cost Comparison

### Scenario: 10,000 transactions

**Pipeline.py:**
- Time: ~2.5 minutes
- Cost: $0 (no API calls)
- Throughput: 4,000 transactions/min

**LLM Agents:**
- Time: ~5-14 hours (at 2-5 sec/transaction)
- Cost: ~$10-100 (at $0.001-0.01/transaction)
- Throughput: 12-30 transactions/min

**Hybrid (95% pipeline, 5% agents):**
- Time: ~3 minutes
- Cost: ~$0.50-5 (only 500 transactions use LLM)
- Best of both worlds!

---

## 📝 Code Examples

### Example 1: Production Batch Processing
```python
# Use pipeline.py
from pipeline import process_transactions_file

# Process 50,000 transactions
results = process_transactions_file(
    input_file='data/monthly_transactions.csv',
    output_file='output/categorized_monthly.csv',
    batch_size=200
)
# Fast, free, scalable
```

### Example 2: Interactive Debugging
```python
# Use LLM Agent
from agents.retriever_agent import create_retriever_agent

agent = create_retriever_agent()
response = agent.run(
    "Why was 'AMAZON PAY INDIA TXN 12345' categorized as Shopping? "
    "Show me the similar merchants found and explain the reasoning."
)
print(response.content)
# Rich explanation with LLM reasoning
```

### Example 3: Hybrid Approach
```python
from pipeline import process_transactions_file
from agents.retriever_agent import create_retriever_agent

# Step 1: Bulk process (fast)
results = process_transactions_file('data/transactions.csv')

# Step 2: Identify edge cases
edge_cases = results[
    (results['match_quality'] == 'low') | 
    (results['confidence_score'] < 0.70)
]

# Step 3: Use agent for explanations (only for edge cases)
if len(edge_cases) > 0:
    agent = create_retriever_agent()
    explanations = []
    
    for _, row in edge_cases.iterrows():
        explanation = agent.run(
            f"Analyze this transaction: {row['original_description']}. "
            f"Why was it categorized as {row['category']} with confidence {row['confidence_score']:.2f}?"
        )
        explanations.append(explanation.content)
    
    # Save explanations for review
    edge_cases['llm_explanation'] = explanations
```

---

## 🎯 Decision Flowchart

```
Start: Need to process transactions
│
├─ Processing 100+ transactions?
│  │
│  ├─ YES → Use pipeline.py
│  │         (Fast, free, scalable)
│  │
│  └─ NO → Continue...
│
├─ Need explanations/reasoning?
│  │
│  ├─ YES → Use LLM Agents
│  │         (Rich explanations)
│  │
│  └─ NO → Continue...
│
├─ Production/batch processing?
│  │
│  ├─ YES → Use pipeline.py
│  │         (Reliable, fast)
│  │
│  └─ NO → Continue...
│
└─ Interactive/debugging?
   │
   └─ YES → Use LLM Agents
             (Reasoning, explanations)
```

---

## 📋 Summary

| Use Case | Tool | Why |
|----------|------|-----|
| Process 10,000 transactions | `pipeline.py` | Fast, free, scalable |
| Understand why transaction categorized | LLM Agents | Rich explanations |
| Production batch job | `pipeline.py` | Reliable, deterministic |
| Debug edge case | LLM Agents | Reasoning, suggestions |
| User asks "Why Shopping?" | LLM Agents | User-friendly explanation |
| Monthly categorization | `pipeline.py` | Cost-effective, fast |
| Low-confidence transaction | LLM Agents | Better reasoning |
| Interactive demo | LLM Agents | Engaging explanations |

---

## 🚀 Quick Reference

```bash
# Batch processing (production)
python pipeline.py --input data/transactions.csv

# Interactive testing (development)
python agents/preprocessor_agent.py
python agents/retriever_agent.py

# Hybrid approach (best practice)
# 1. Process bulk with pipeline
# 2. Use agents for edge cases only
```

---

**Remember:** Pipeline for speed and scale, Agents for reasoning and explanations!

