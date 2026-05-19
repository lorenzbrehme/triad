# Experiments

This folder contains the experimental datasets, scripts, and evaluation outputs used for the TRIAD experiments.


# Directory Structure

## `qa_sets/`

Generated QA datasets created with the TRIAD pipeline.

### Contents

* `[dataset]_validated.json`
  Raw generated QA sets before filtering.

* `[dataset]_final/`
  Filtered QA pairs that passed validation.

* `[dataset]_extraction_qa/`
  QA items paired with their extracted contextlist.

### `human_validation/`

Files related to manual QA validation and annotation.

#### Contents

* Sampling notebook (`select_for_human_validation.ipynb`)
  Performs sampling of QA items for human evaluation.

* `[dataset]_sample.json`
  Sampled QA items prepared for annotators.

* `validator_[dataset]_sample.json`
  Validation samples assigned to annotators.

* `human_validation.xlsx`
  Human annotation results and evaluation labels.

* `viewer.html`
  Lightweight viewer for browsing sampled QA items and annotations.

---

## `static_qas/`

Questions sourced from static benchmark datasets used for evaluation.

---

## `run_questions/`

Scripts used to run the RAG retrieval and generation pipeline across different experimental configurations.

---

## `rag_answer_data/`

Collected outputs from all tested RAG system configurations.

### Naming Convention

Files follow the schema:

```text
dataset_llm_numberchunks_embeddingmodel.json
```

### Examples

```text
hotpot_gpt-5-nano_5_chunks-bge-small-rag.json
mhqa_hotpot_5-nano_5_chunks-bge-small-rag.json
```

Where:

* `dataset` → QA dataset used, mhqa is the TRIAD generated Dataset
* `llm` → generation model
* `numberchunks` → number of retrieved chunks
* `embeddingmodel` → embedding model used for retrieval

### Additional Folder

* `not_answerable/`
  Contains results for questions determined to be unanswerable given the non relevant context for a specific RAG run.

---

## `evaluation_pipeline/`

Scripts and outputs for evaluating RAG responses.

### Contents

* `evaluation.ipynb`
  Notebook used to evaluate generated answers with RAGAS.

* `evaluation_results_[rag_answer_set_name].json`
  Stored evaluation outputs and metrics.

---

## `rag_to_be_tested/`

RAG system implementation used during evaluation experiments.

### Running the FastAPI Application

```bash
cd rag_to_be_tested/
uvicorn main:app --reload
```

---

# Experiment Workflow Overview

1. Generate QA datasets with TRIAD, including valideation and extraction
2. Run RAG systems on QA datasets
3. Collect generated answers
4. Evaluate outputs using RAGAS

---

# Notes

* Generated QA datasets and static benchmark datasets are evaluated separately.
* Human validation is used to assess QA quality beyond automated checks.
* File naming conventions are important for reproducibility and experiment tracking.
