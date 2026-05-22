# TRIAD — A Three-Stage Automated Multi-Hop Dataset Generation Approach

A comprehensive system for generating, validating and extracting question-answer pairs for RAG (Retrieval-Augmented Generation) systems. TRIAD leverages multiple LLM models to create high-quality QA datasets from complex document collections.

![Pipeline](images/mhqa.png)

## Table of Contents

- [Project Overview](#-project-overview)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Components](#-components)
- [Supported Datasets](#-supported-datasets)
- [Models](#-models)

## Project Overview

TRIAD is designed to:
- **Generate** synthetic QA pairs from document collections using various question types
- **Validate** generated QA pairs for quality assurance
- **Extract** relevant information from QA datasets
- **Work with domain specific datasets** 


## Directory Structure

```
triad/
├── README.md                          # This file
├── main.ipynb                         # Main execution notebook
├── .env.example                       # Environment variables template
│
├── qa_generator/                      # Question and Answer Generation
│   ├── generator.py                   # Core QA generation logic
│   └── prompt_templates.py            # LLM prompt templates
│
├── qa_classifier/                     # Question Classification
│   └── classifier.py                  # Classification logic (temporal/non-temporal)
│
├── qa_extractor/                      # QA Context List Extraction
│   └── extractor.py                   # Context List extraction utilities
│
├── qa_validator/                      # QA Validation & Quality Assurance
│   └── validator.py                   # Validation logic
│
├── Models/                            # LLM Model Interfaces
│   ├── chat_gpt.py                    # OpenAI ChatGPT integration
│   ├── gemini.py                      # Google Gemini integration
│   └── request.py                     # Generic LLM request handler
│
├── Experiments/                       # Experimental Workflows
│   ├── qa_sets                        # Generated QA sets
|   ├── preprocessing_hotpot.ipynb     # HotPot dataset preprocessing
│   ├── preprocessing_musique.ipynb    # MusiQue dataset preprocessing
│   ├── evaluation_pipeline/           # Evaluation scripts and pipelines
│   ├── rag_to_be_tested/              # RAG systems to test
│   ├── rag_answer_data/               # Generated answers from RAG
│   ├── run_questions/                 # Question execution
│   ├── static_qas/                    # Static QA datasets
│   └── README.md                      # Details for Experiments
|
└── database/                          # Vector Database Setup
    ├── hotpot_vectordatabase.ipynb    # HotPot vector DB initialization
    └── musique_vectordatabase.ipynb   # MusiQue vector DB initialization

```

## Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL database (for vector storage)
- API keys for LLM services (OpenAI, Google)

### Step 1: Clone and Environment Setup

```bash
git clone [link]
```


### Step 2: Set Up Environment Variables

Copy and configure the environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:


| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT access | Yes (if using ChatGPT) |
| `GOOGLE_API_KEY` | Google API key for Gemini access | Yes (if using Gemini) |
| `LANGSMITH_TRACING` | Enable LangSmith tracing (true/false) | No |
| `LANGSMITH_API_KEY` | LangSmith API key for monitoring | No |

### Step 4: Set Up PostgreSQL Vector Database

```bash
# Run the database initialization notebooks
# For HotPot:
jupyter notebook database/hotpot_vectordatabase.ipynb

# For MusiQue:
jupyter notebook database/musique_vectordatabase.ipynb
```

Default connection: `postgresql+psycopg://postgres:password@localhost:5432/hotpot`



## Usage

### Main Workflow

The primary workflow is in `main.ipynb`:

```python
# Load embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://postgres:password@localhost:5432/hotpot"
)

# Initialize generator with LLM model
qa_generator = QA_Generator(model=gemini_model, vector_store=vector_store)

# Generate QA pairs for different types
setups = [
    {"qa_type": "comparison_question", "binary": True, "count": 100},
    {"qa_type": "intersection_question", "binary": False, "count": 100},
    {"qa_type": "attribute_composition", "binary": True, "count": 100}
]

# Process each setup
for setup in setups:
    # Generate QA pairs
    # Validate results
    # Extract and save data
```

### Using Individual Components

#### 1. Generate QA Pairs

```python
from qa_generator.generator import QA_Generator

generator = QA_Generator(model=your_model, vector_store=vector_store)
question = generator.generate_question(chunk, qa_type, binary,role)
```



#### 2. Validate QA Pairs

```python
from qa_validator.validator import QA_Validator

validator = QA_Validator(model=your_model)
validated_qa = validator.validate_qa_set(question)

#loop for validation
if validated_qa.get('accepted') is not True:
    question = generator.feedback_loop(validated_qa.get('reason'), question['chunks'], question['bridging_topic'])
    # do the validation again


```

#### 3. Extract QA Data

```python
from qa_extractor.extractor import QA_Extractor

extractor = QA_Extractor(model = your_model, vector_store=vector_store)
extracted_contexts = extractor.extract_contexts(validated_qa)
```




