# Automated GitHub Issue Classification Using Machine Learning

## Project Overview

The exponential growth of open-source software has led to an increase in issues reported on platforms like GitHub, spanning categories such as bug reports, feature requests, and general inquiries. Efficiently triaging these issues is critical for software project management. This project explores automated classification of GitHub issues using machine learning (ML) and deep learning (DL) techniques, leveraging transformer-based embeddings and a range of classification models.

### Key Features
- **Framework Development**: Combines contextual language embeddings with classifiers.
- **Transformer Models**: Uses state-of-the-art embeddings from RoBERTa and Sentence Transformers.
- **Classifiers**: Evaluates SVM, Random Forest, LightGBM, XGBoost, LSTM, and DistilBERT models.
- **Dataset**: Employs a balanced dataset of 3,000 GitHub issues from repositories like React, TensorFlow, and OpenCV.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Implementation](#implementation)
5. [Results](#results)
6. [Future Work](#future-work)
7. [References](#references)

---

## Introduction

This project addresses the challenges of manual issue triaging by introducing an automated framework utilizing ML and DL models. It highlights the trade-offs between accuracy and computational efficiency, emphasizing the transformative potential of pre-trained transformers for issue classification.

---

## Dataset

- **Source**: GitHub repositories, including `facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`, and `opencv/opencv`.
- **Size**: 3,000 issues evenly distributed across categories (bug, feature, question).
- **Features**:
  - Repository Name
  - Creation Date
  - Issue Title
  - Issue Body
  - Label (Bug, Feature, Question)

---

## Methodology

1. **Preprocessing**:
   - Removes code snippets, URLs, and unnecessary text.
   - Normalizes and cleans the dataset.
2. **Feature Extraction**:
   - Leverages embeddings from RoBERTa and Sentence Transformers.
   - Combines Title and Body features into a unified feature matrix.
3. **Classification Models**:
   - Traditional ML: SVM, Random Forest, LightGBM, XGBoost.
   - Deep Learning: LSTM, DistilBERT with Adapters.

---

## Implementation

- **Frameworks & Tools**:
  - [HuggingFace Transformers](https://huggingface.co/)
  - PyTorch
  - scikit-learn
- **Steps**:
  - Preprocess data and generate embeddings.
  - Train models using grid-search hyperparameter tuning.
  - Evaluate performance using F1-score, Accuracy, Precision, and Recall.

---

## Results

- **Top Performer**: DistilBERT (F1-score: 0.992).
- **Insights**:
  - Transformer models significantly outperform traditional classifiers.
  - Embedding combination of RoBERTa (Body) and Sentence Transformer (Title) yields the best results.
  - Trade-offs between computational efficiency (LightGBM) and accuracy (DistilBERT).

---

## Future Work

- Expand dataset diversity and size.
- Fine-tune transformer embeddings for domain-specific tasks.
- Explore state-of-the-art architectures like GPT and T5.
- Develop a lightweight, real-time issue classification tool.

---

## References

- See the full list of references in the [project report](https://github.com/tan-pixel/Automated-Issue-Classification).
