# DocIE-Probing Repository Overview

This repository contains code and data for the paper "Probing Representations for Document-level Event Extraction" (EMNLP 2023 Findings). 

The project focuses on analyzing different models' capabilities in document-level event extraction.

**Code Safety Warning**: Some models use outdated dependencies (like older transformers), which has critical vulnerabilities. But for compatibility purposes this repo will not update them. Make sure to use an isolated, separate environment.

## Repository Structure

### 1. Corpora
Contains two main datasets, with both converted for different models:

- **MUC 3/4 (Message Understanding Conference)**: 
  - Historical dataset from the 1990s
  - Contains various versions including:
    - Base MUC dataset
    - Two attempts at adding triggers, which are [expanded in later works](https://arxiv.org/abs/2411.08708v1). 
  - Includes both full text and sentence-level (SentCat) variants
  - Split into 200 test, 200 dev, 1300 training examples

- **WikiEvents**:
  - Similar to MUC, but have 20/20/206 test/dev/train split.

### 2. Models
The repository implements several event extraction models:

- [**DyGIE++**](https://github.com/dwadden/dygiepp): A document-level information extraction system
  - Includes configuration files and training scripts
  - Contains output processing and error analysis tools

- [**GTT**](https://github.com/xinyadu/gtt): 
  - Template filling with generative transformers
  - Includes evaluation scripts and utilities

- [**TANL**](https://github.com/amazon-science/tanl): 
  - Text-to-text approach for information extraction
  - Contains pre/post processing scripts and evaluation tools

- **Naive BERT Baseline**

Models are modified to take environmental variables to save the embedding they used. More elegant solutions could use [transformer_lens](https://github.com/TransformerLensOrg/TransformerLens) or torch hook.

### 3. Probing
Contains code and data for analyzing model representations:

- **Code/**:
  - `getX_embeddings.py`: Extracts embeddings from different models
  - `getY_labels.py`: Processes dataset labels
  - `probing.py`: Main probing implementation
  - `confusion_matrix.py`: Visualization and analysis tools
  - Additional utilities and analysis notebooks

- **X_Embeddings/**: Contains extracted embeddings from different models
  - bfloat16 version included in the Github repo
- **Y_labels/**: Contains processed labels for both MUC and WikiEvents
- **Y_labels_Tokenizer_Specific/**: Model-specific label processing

## Getting Started

1. First, familiarize yourself with the dataset formats in the `Corpora` directory
2. Check the model implementations in the `Model` directory
3. Use the probing tools in the `Probing` directory to analyze model representations

## Key Features

- Multiple model implementations for document-level event extraction
- Comprehensive probing framework for analyzing model representations
- Support for both full text and sentence-level processing
- Tools for data preprocessing, model training, and analysis
