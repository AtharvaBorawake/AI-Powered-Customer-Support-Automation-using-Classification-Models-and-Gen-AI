# AI-Powered Customer Support Automation using Classification Models and Generative AI

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-ff4b4b.svg)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5%20Flash-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

# Overview

This project presents an **end-to-end AI-powered Customer Support Automation System** that automatically classifies customer support tickets into the appropriate support queue and generates professional acknowledgment responses using **Google Gemini 2.5 Flash**.

The system combines:

- Traditional Machine Learning Models
- Deep Learning (BiLSTM + Attention)
- Transformer Models (DistilBERT)
- Generative AI (Gemini API)
- Interactive Streamlit Dashboard

The objective is to reduce manual ticket triaging, improve response times, increase support team productivity, and enhance customer satisfaction.

---

# Business Problem

Organizations receive thousands of customer support tickets every day.

Manual ticket classification introduces:

- Delayed ticket assignment
- Incorrect queue routing
- Increased operational costs
- Poor customer experience
- Reduced support agent productivity

This project automates the complete workflow:

**Customer Ticket → Queue Prediction → Automated Customer Response**

---

# Business Use Cases

### 1. Customer Support Automation

Automatically classify incoming support tickets into appropriate departments.

### 2. Faster Ticket Resolution

Reduce time spent manually reviewing and assigning tickets.

### 3. Automated Customer Acknowledgment

Generate immediate professional responses to customers.

### 4. Intelligent Ticket Routing

Route tickets to the correct queue automatically.

### 5. Priority Detection

Identify urgent tickets using ticket content.

### 6. Operational Cost Reduction

Reduce manual effort involved in support ticket triaging.

### 7. Improved Customer Experience

Provide faster and more consistent responses.

### 8. Agent Productivity Enhancement

Allow support agents to focus on solving issues.

### 9. Support Data Insights

Analyze ticket trends and queue distributions.

### 10. Scalable Support Infrastructure

Handle large ticket volumes efficiently.

---

# Dataset

## Source

Hugging Face Dataset:

**Tobi-Bueck/customer-support-tickets**

Dataset Link:

https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets

---

## Dataset Fields

| Column | Description |
|----------|-------------|
| subject | Ticket subject |
| body | Customer message |
| queue | Target support department |
| priority | Ticket priority |
| tags | Ticket tags |
| language | Language of ticket |

---

## Target Variable

```text
queue
```

Examples:

- Billing and Payments
- Technical Support
- IT Support
- Customer Service
- Human Resources
- Product Support
- Returns and Exchanges
- General Inquiry
- Sales and Pre-Sales
- Service Outages and Maintenance

---

# System Architecture

<p align="center">
  <img src="assets/diagram%20customer%20support.png" alt="Customer Support Architecture" width="1000">
</p>

```text
Customer Ticket
        │
        ▼
Data Loading
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Classification Models
 ├── Logistic Regression
 ├── Naive Bayes
 ├── Linear SVM
 ├── BiLSTM + Attention
 └── DistilBERT
        │
        ▼
Predicted Queue
        │
        ▼
Gemini 2.5 Flash API
        │
        ▼
Professional Acknowledgment Response
        │
        ▼
Streamlit Dashboard
```


---

# Project Structure

```text
AI-Powered-Customer-Support-Automation/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── assets/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── baseline_models.py
│   ├── train_lstm.py
│   ├── train_distilbert.py
│   ├── predictor.py
│   └── gemini_client.py
│
├── models/
│
├── saved_models/
│
└── outputs/
```

---

# Data Preprocessing

The preprocessing pipeline performs:

### Classical Machine Learning Models

- Lowercasing
- URL Removal
- Email Removal
- HTML Removal
- Stopword Removal
- Lemmatization
- Special Character Removal

### Transformer Models

Minimal preprocessing:

- Whitespace normalization
- Preserve original context
- DistilBERT tokenization

---

# Feature Engineering

Additional features considered:

### Text-Based Features

- TF-IDF Features
- N-Grams (1–3)

### Business Features

- Keyword Presence
- Urgency Indicators
- Ticket Length
- Priority Information

Examples:

```text
urgent
asap
immediately
refund
payment failed
login issue
```

---

# Models Implemented

---

## 1. Logistic Regression

Baseline machine learning classifier.

Features:

- TF-IDF
- N-Grams

Advantages:

- Fast training
- Good interpretability

---

## 2. Multinomial Naive Bayes

Probabilistic text classification model.

Advantages:

- Extremely fast
- Strong baseline for NLP

Limitations:

- Assumes feature independence

---

## 3. Linear SVM

Linear Support Vector Machine.

Advantages:

- Strong performance on text data
- Handles high-dimensional sparse features

---

## 4. BiLSTM with Attention

Deep Learning architecture implemented using PyTorch.

Architecture:

```text
Input Text
      │
      ▼
Embedding Layer
      │
      ▼
BiLSTM
      │
      ▼
Multi-Head Attention
      │
      ▼
Dense Layers
      │
      ▼
Softmax
```

Features:

- Bidirectional Context Learning
- Attention Mechanism
- Layer Normalization
- Dropout Regularization

---

## 5. DistilBERT

Fine-tuned Transformer model.

Architecture:

```text
Input Text
      │
      ▼
Tokenizer
      │
      ▼
DistilBERT Encoder
      │
      ▼
Classification Head
      │
      ▼
Softmax
```

Advantages:

- Context-aware embeddings
- Better semantic understanding
- State-of-the-art NLP performance

---

# Model Training Strategy

### Train-Test Split

```text
80% Training
20% Testing
```

### DistilBERT

```text
Train : 80%
Validation : 10%
Test : 10%
```

### Class Imbalance Handling

- Class Weights
- Balanced Sampling
- Macro F1 Evaluation

### Optimization

- AdamW Optimizer
- Learning Rate Scheduling
- Early Stopping
- Gradient Clipping

---

# Evaluation Metrics

The following metrics were used:

### Accuracy

Measures overall correctness.

### Precision

Measures prediction quality.

### Recall

Measures ability to detect all classes.

### F1 Score

Balances Precision and Recall.

### Macro F1

Important for imbalanced classes.

### Confusion Matrix

Visualizes classification errors.

---

# Experimental Results

| Model | Accuracy | Macro F1 |
|---------|----------|----------|
| Naive Bayes | 42.99% | 23.47% |
| Logistic Regression | 58.51% | 49.13% |
| Linear SVM | 70.84% | 71.96% |
| BiLSTM + Attention | 74.27% | 75.15% |
| DistilBERT | 75.90% | 78.80% |

---

# Best Performing Model

## DistilBERT

### Accuracy

```text
75.90%
```

### Macro F1

```text
78.80%
```

DistilBERT achieved the best balance between:

- Accuracy
- Generalization
- Minority-class performance
- Inference speed

---

# Why Accuracy Is Not Higher

Several real-world challenges affect performance:

### Class Overlap

Examples:

```text
Technical Support
vs
IT Support
```

Many tickets contain similar language.

---

### Class Imbalance

Some queues contain significantly fewer examples.

---

### Ambiguous Tickets

Example:

```text
My account payment failed and now I cannot login.
```

Could belong to:

- Billing
- Technical Support
- Customer Service

---

### Label Noise

Human-created labels may contain inconsistencies.

---

### Long Ticket Truncation

DistilBERT processes limited token lengths.

---

### Domain Variability

Customers use different writing styles.

---

# Gemini Integration

After classification:

```text
Ticket Text
      +
Predicted Queue
      │
      ▼
Gemini 2.5 Flash
      │
      ▼
Professional Customer Response
```

---

## Example

### Input Ticket

```text
I was charged twice for my subscription.
```

### Predicted Queue

```text
Billing and Payments
```

### Gemini Response

```text
Dear Customer,

Thank you for contacting us.

We have received your request and routed it to our Billing and Payments team for review.

Our team will investigate the issue and assist you as soon as possible.

Kind regards,
Customer Support Team
```

---

# Streamlit Application

The application provides:

### Dashboard

- Project Overview
- Model Performance

### Ticket Classification

- Enter support ticket
- Predict queue

### AI Response Generation

- Generate Gemini response

### Model Comparison

- Compare all models

### Error Analysis

- Understand limitations

---

# Installation

## Clone Repository

```bash
git clone https://github.com/AtharvaBorawake/AI-Powered-Customer-Support-Automation-using-Classification-Models-and-Gen-AI.git

cd AI-Powered-Customer-Support-Automation-using-Classification-Models-and-Gen-AI
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Gemini API

Create:

```text
.env
```

Add:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

# Reproducing Results

## Train Baseline Models

```bash
python src/baseline_models.py
```

---

## Train BiLSTM

```bash
python src/train_lstm.py
```

---

## Train DistilBERT

```bash
python src/train_distilbert.py
```

---

# Run Streamlit Application

```bash
streamlit run app.py
```

---

# Model Files

Due to GitHub storage limitations, trained model files are excluded from this repository.

Excluded folders:

```text
models/
saved_models/
```

To regenerate the artifacts, run the training scripts provided in the repository.

---

# Future Enhancements

- RoBERTa
- DeBERTa-v3
- Ensemble Learning
- Sentiment Analysis
- Priority Prediction
- Multi-Label Classification
- Active Learning
- Human-in-the-Loop Feedback
- RAG-based Support Systems

---

# Technology Stack

### Programming

- Python

### Machine Learning

- Scikit-Learn

### Deep Learning

- PyTorch

### NLP

- NLTK
- Hugging Face Transformers

### Generative AI

- Google Gemini 2.5 Flash

### Deployment

- Streamlit

### Dataset

- Hugging Face Datasets

---

# Key Learnings

- End-to-end NLP pipeline design
- Text preprocessing techniques
- Classical ML vs Deep Learning comparison
- Transformer fine-tuning
- Generative AI integration
- Model evaluation and error analysis
- Streamlit deployment

---

# Author

**Atharva Borawake**

AI-Powered Customer Support Automation using Classification Models and Generative AI

Built using NLP, Deep Learning, Transformers, and Generative AI.

---

# License

This project is licensed under the MIT License.
