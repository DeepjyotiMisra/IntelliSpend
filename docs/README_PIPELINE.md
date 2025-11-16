# IntelliSpend Pipeline - Processing Raw Transactions

## Overview

The main pipeline (`pipeline.py`) processes raw transaction CSV files through the IntelliSpend system, categorizing each transaction using the Preprocessor and Retriever agents.

## Input Format

The pipeline expects a CSV file with the following columns:
- `date`: Transaction date (optional)
- `amount`: Transaction amount (optional)
- `description`: Transaction description (required)

Example: `data/raw_transactions.csv`

## Usage

### Basic Usage

```bash
python pipeline.py
```

This will:
- Load `data/raw_transactions.csv`
- Process all transactions
- Save results to `output/categorized_transactions.csv`

### Custom Input/Output

```bash
python pipeline.py --input data/my_transactions.csv --output output/my_results.csv
```

### Custom Column Names

```bash
python pipeline.py --description-col "transaction_desc" --amount-col "value"
```

### Batch Processing

```bash
python pipeline.py --batch-size 200
```

## Output Format

The output CSV includes all original columns plus:

- `merchant`: Identified merchant name
- `category`: Assigned category
- `confidence_score`: Similarity score (0-1)
- `payment_mode`: Extracted payment mode (UPI, NEFT, IMPS, CARD, etc.)
- `num_matches`: Number of similar merchants found
- `retrieval_source`: Source of retrieval (faiss, none, error)
- `processing_status`: Processing status (success, error)

## Processing Flow

```
Raw Transaction CSV
    ↓
Load Transactions
    ↓
For each transaction:
    ├─ Preprocessor Agent
    │  └─ Normalize description
    │  └─ Extract payment mode
    │
    └─ Retriever Agent
       └─ Find similar merchants
       └─ Get category from best match
    ↓
Save Categorized Results
```

## Statistics

After processing, the pipeline prints:
- Total transactions processed
- Success/error counts
- Category distribution
- Confidence score statistics

## Example Output

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

📊 Category Distribution:
   Shopping: 450 (22.5%)
   Transport: 320 (16.0%)
   Food & Dining: 280 (14.0%)
   ...

📊 Confidence Statistics:
   Average confidence: 0.847
   High confidence (≥0.85): 1200 (60.0%)
   Medium confidence (0.70-0.85): 600 (30.0%)
   Low confidence (<0.70): 201 (10.0%)
```

## Prerequisites

1. **FAISS Index Built**: 
   ```bash
   python utils/faiss_index_builder.py
   ```

2. **Environment Configured**: 
   - `.env` file with OpenAI/Azure credentials
   - `config/taxonomy.json` exists

3. **Dependencies Installed**:
   ```bash
   pip install -r requirements.txt
   ```

## Performance

- **Batch Processing**: Processes transactions in batches for efficiency
- **Progress Tracking**: Shows progress bar during processing
- **Error Handling**: Continues processing even if individual transactions fail
- **Logging**: Detailed logs saved to `logs/intellispend.log`

## Next Steps

After processing, you can:
1. Review `output/categorized_transactions.csv`
2. Filter low-confidence predictions for review
3. Use results for evaluation and metrics
4. Feed back corrections to improve the system

