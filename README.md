# CE-RAG4EM: Cost-Efficient RAG for Entity Matching with LLMs

This repository provides the source code, data, and supplemental material  of our paper "CE-RAG4EM: Cost-efficient RAG for Entity Matching with LLMs". 

## Introduction

CE-RAG4EM is a cost-efficient RAG for entity matching that reduces computation through blocking-based batch retrieval and generation.

- Introduce a blocking strategy to reduce the overall cost of context retrieval and LLM inference for entity matching
- Retrieves and searches relevant context from external knowledge graphs across domains (e.g., Wikidata, KG20C)
- Augment LLMs for entity matching with retrieved and refined context from an external knowledge base
- Supports multiple entity matching datasets (abt, amgo, beer, dbac, dbgo, foza, itam, waam, wdc)

## Datasets

The project supports multiple entity matching benchmark datasets:

- **abt**: Abt-Buy dataset
- **amgo**: Amazon-Google dataset
- **beer**: Beer dataset
- **dbac**: DBLP-ACM dataset
- **dbgo**: DBLP-GoogleScholar dataset
- **foza**: Fodors-Zagats dataset
- **itam**: iTunes-Amazon dataset
- **waam**: Walmart-Amazon dataset
- **wdc**: Web Data Commons dataset

Place your raw datasets in the `data/raw/` directory.

## Quick Start

### 1. Environment Setup
The project includes a `rag4em.yml` file that contains all necessary dependencies, and creates a conda environment from YAML file.

```bash
# Create conda environment from the YAML file
conda env create -f rag4em.yml

# Activate the environment
conda activate rag4em
```

### 2. Configure LLM APIs

#### For OpenAI GPT Models

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### For Google Gemini Models

Set your Google API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

#### For Hugging Face Models

Login with your Hugging Face token (required for gated models):

```bash
huggingface-cli login
```

### 3. Configure Vector AstraDB 

#### Train emebddings and upload to AstraDB

- Create "db_kg20c" in AstraDB
- Set your AstraDB API endpoint and AstraDB Application API key:

```bash
export ASTRA_DB_API_ENDPOINT="your-astradb-api-endpoint-here"
export ASTRA_DB_APPLICATION_TOKEN="your-astradb-application-api-key"
```

### 4. Run the Method

### End-to-End Pipeline from Blocking to Matching

```bash
# Step 1: Generate blocking pairs
python blocking_pair_generation.py -d abt -p test 

# Step 2: Retrieve contextual knowledge per block
python batch_retrieval.py -d abt -p test -b QG -maxb 6 -kg wikidata

# Step 3: Run CE-RAG for knowledge-augmented inference for entity matching
python ce_rag4em_main.py -d abt -p test -m gpt-4o-mini -b QG -maxb 6 -kg wikidata
```

**Key Arguments:**
- `-d`: Dataset to use (abt, amgo, beer, dbac, dbgo, foza, itam, waam, wdc)
- `-p`: Data partition (train, test, valid)
- `-m`: LLM model to use (gpt-4o-mini, qwen3-4b, etc.)
- `-b`: Blocking method to use (SB, QG, EQG, SA, ESA)
- `-maxb`: Maximum blocking size to process for batch retreival and inference
- `-kg`: KG source for RAG (wikidata, kg20c)


## Usage Examples

### Example 1: LLM4EM with GPT-4o-mini
```bash
# Step 1: Configure context_conifg in the `ce_rag4em_main.yml`
context_config = {
        "enabled": False,  # Set to False to disable context retrieval
        "context_type": "qid",  # "pid", "qid", or "triple"
        "top_k": 2   # Number of top retrieval results to use (1 or 2)
}

# Setep 2: Run the main python file
python ce_rag4em_main.py -d abt -p test -m gpt-4o-mini -b QG -maxb 6 -kg wikidata
```

### Example 2: RAG4EM with Top-1 QID triple and Gemini-2.0-flash-lite

```bash
# Step 1: Configure context_conifg in the `ce_rag4em_main.py`
context_config = {
        "enabled": True,  # Set to False to disable context retrieval
        "context_type": "qid",  # "pid", "qid", or "triple"
        "top_k": 1   # Number of top retrieval results to use (1 or 2)
}

# Setep 2: Run the main python file
python ce_rag4em_main.py -d abt -p test -m gemini-2.0-flash-lite -b QG -maxb 6 -kg wikidata
```

### Example 3: KG-RAG4EM with Top-2 BFS triple and Qwen3-4b

```bash
# Step 1: Configure context in the `ce_rag4em_main.py`
context_config = {
        "enabled": True,  # Set to False to disable context retrieval
        "context_type": "triple",  # "pid", "qid", or "triple"
        "top_k": 2   # Number of top retrieval results to use (1 or 2)
}
# Step 2: Configure tirple in the `ce_rag4em_main.py` if context_type is "triple"
    triple_id_type = "QID"  # "QID" or "PID", 
    triple_generation_type = "BFS"  # "BFS" or "EXP (expansion)" Triple search approach for triple generation
    top_k_entities = 3  # Number of top entities/properties to use for triple generation

# Setep 3: Run the main python file
python ce_rag4em_main.py -d abt -p test -m qwen3-4b QG -maxb 6 -kg wikidata
```

### Example 4: KG-RAG4EM with Top-2 EXP triple from KG20C and Qwen3-4b

```bash
# Step 1: Configure context in the `ce_rag4em_main.py`
context_config = {
        "enabled": True,  # Set to False to disable context retrieval
        "context_type": "triple",  # "pid", "qid", or "triple"
        "top_k": 2   # Number of top retrieval results to use (1 or 2)
}
# Step 2: Configure tirple in the `ce_rag4em_main.py` if context_type is "triple"
    triple_id_type = "QID"  # "QID" or "PID", 
    triple_generation_type = "EXP"  # "BFS" or "EXP (expansion)" Triple search approach for triple generation
    top_k_entities = 3  # Number of top entities/properties to use for triple generation

# Setep 3: Run the main python file
python ce_rag4em_main.py -d abt -p test -m qwen3-4b QG -maxb 6 -kg kg20c
```

## Output

The system generates several types of outputs:

1. **Blocking outputs** (`blocking_outputs/`): Candidate entity pairs generated by blocking methods
2. **Retrieval outputs** (`retrieval_outputs/`): Retrieved context from knowledge graphs
3. **Outputs** (`output/`): The prompt with different retrieved contexts, output of LLM inference, and the final results with evaluation metrics
4. **Logs** (`logs/`): Detailed execution logs for further analysis


## Acknowledgment

### Dataset
The abt, amgo, beer, dbac, dbgo, foza, itam, waam datasets and the wdc dataset originated from the following works:
```
Deep Learning for Entity Matching: A Design Space Exploration
https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md

SC-Block: Supervised Contrastive Blocking Within Entity Resolution Pipelines
https://webdatacommons.org/largescaleproductcorpus/wdc-block/
```

### KG

The domain-specific KG (KG20C) originated from the following work:
```
KG20C & KG20C-QA: Scholarly Knowledge Graph Benchmarks for Link Prediction and Question Answering
https://github.com/tranhungnghiep/KG20C
```

we thank them for sharing the dataset and KG.

### VectorDB
The Wikidata VectorDB and its API access are provided by the team behind the [Wikidata Embedding Project](https://www.wikidata.org/wiki/Wikidata:Embedding_Project). We thank them for creating and maintaining this excellent project.
