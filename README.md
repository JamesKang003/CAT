# CAT: Clinical-Aware Transformer

A biomedical language model and data pipeline designed to ensure **data integrity, reliability, and predictable behavior** when processing complex clinical-style text.

---

## Overview

CAT is a transformer-based language model trained on **PubMed Central (PMC) Open Access full-text biomedical articles**.  

Unlike typical LLM projects focused on model performance, this project focuses on:

- **Data integrity**
- **Reliable evaluation**
- **Failure prevention in real-world pipelines**

The system was built from the ground up, including:

- XML data ingestion
- Tokenizer design and debugging
- Training pipeline architecture
- Evaluation correctness (data leakage prevention)

---

## Motivation

In healthcare systems, **incorrect interpretation of data can lead to real-world consequences**.

Most ML pipelines focus on improving metrics (e.g., perplexity), but:
- Metrics can be misleading
- Evaluation can be contaminated
- Systems can appear “accurate” while being fundamentally unreliable

This project focuses on:

> **Ensuring that model outputs are based on valid generalization, not artifacts of flawed data pipelines**

---

## Key Contributions

### 1. Data Integrity & Leakage Prevention

**Problem:**
- Documents were split across train/validation/test sets
- This introduced **data leakage**
- Result: artificially inflated performance

**Solution:**
- Implemented **article-level splitting**
- Ensured no document appears across multiple splits

**Impact:**
- Eliminated contamination between datasets
- Produced **trustworthy evaluation metrics**

---

### 2. XML Data Ingestion Pipeline

**Problem:**
- PMC data is stored in XML/NXML format
- Contains:
  - tags
  - citations
  - structural noise

**Solution:**
- Built `data_ingestion.py` pipeline
- Parsed raw XML into clean text
- Removed structural artifacts
- Preserved meaningful biomedical context

---

### 3. Tokenizer Debugging & Vocabulary Scaling

**Problem:**
- Byte-level artifacts (e.g., `Ġ`) appeared during decoding
- Biomedical terms were fragmented
- Reduced semantic clarity

**Solution:**
- Added **ByteLevel decoder explicitly**
- Expanded vocabulary:
  - 16K → 32K tokens

**Impact:**
- Improved token reconstruction
- Preserved domain-specific terminology

---

### 4. Context & Data Restructuring

**Problem:**
- Full-document training caused:
  - low information density
  - inefficient learning

**Solution:**
- Chunked data into **paragraph-level segments**
- Inserted structured boundaries (`[SEP]`)
- Increased context window:
  - 256 → 512

**Impact:**
- Improved learning efficiency
- Enabled better contextual reasoning

---

## Results

| Metric | Before | After |
|------|--------|-------|
| Validation Perplexity | 34,492 | 47.11 |
| Cross-Entropy | ~10.45 | ~3.95 |
| Reduction in CE | - | **~62% ↓** |
| Repetition (Trigram) | 0.1038 | 0.0510 |
| Reduction in repetition | - | **~51% ↓** |

### Key Insight

> Performance improvements are meaningful **only because evaluation is now valid**.

Without fixing data leakage:
- Lower perplexity would not indicate real model capability

---

## Architecture
Raw PMC XML
↓
Data Ingestion (XML Parsing, Cleaning)
↓
Tokenizer (Byte-Level + Custom Vocab)
↓
Chunking & Structuring
↓
Training Pipeline (PyTorch)
↓
Evaluation (Leakage-Free Split)

---

## Tech Stack

- **Language:** Python  
- **Framework:** PyTorch  
- **Tokenizer:** Custom Byte-Level BPE  
- **Data:** PMC Open Access (XML)  
- **Environment:** CUDA (GPU training)

---

## Key Lessons

### 1. Metrics can lie

Without proper data separation:
- Models appear to improve
- But fail in real-world scenarios

---

### 2. Data pipeline > Model architecture

Most improvements came from:
- fixing ingestion
- fixing splitting
- fixing tokenization

—not changing the model itself

---

### 3. Reliability is a system property

Correctness is not just:
- model weights
- loss functions

It depends on:
- data flow
- preprocessing
- evaluation design

---

## Future Work

- Fine-tuning on structured clinical datasets
- Improving factual consistency
- Integrating retrieval-based grounding (RAG)
- Reducing hallucination in generation

---

## Repository Structure

CAT/
├── scripts/
│ ├── train.py
│ ├── train_resume.py
│ ├── train_tokenizer.py
│ ├── prepare_data.py
│
├── models/
│ ├── tokenizer.json
│ ├── checkpoints/
│
├── data/
│ ├── processed/
│
└── README.md

---

## Summary

CAT is not just a language model.

It is a system designed to answer:

> **Can we trust the outputs of our model?**

By focusing on:
- data integrity
- evaluation correctness
- system-level reliability

this project demonstrates how engineering decisions directly impact the trustworthiness of machine learning systems.
