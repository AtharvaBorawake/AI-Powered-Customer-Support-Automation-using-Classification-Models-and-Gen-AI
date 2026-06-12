
# app.py

import os
import joblib
import pandas as pd
import streamlit as st
import torch

from dotenv import load_dotenv

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from src.gemini_client import GeminiClient


# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Support Automation",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# LOAD DISTILBERT
# =====================================================

@st.cache_resource
def load_model():

    model_path = "saved_models/distilbert_model"

    model = DistilBertForSequenceClassification.from_pretrained(
        model_path
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_path
    )

    encoder = joblib.load(
        "saved_models/label_encoder.pkl"
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer, encoder


model, tokenizer, encoder = load_model()


# =====================================================
# LOAD GEMINI
# =====================================================

@st.cache_resource
def load_gemini():
    return GeminiClient()


gemini = load_gemini()


# =====================================================
# PREDICTION
# =====================================================

def predict_queue(text):

    encoded = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pred = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    queue = encoder.inverse_transform(
        [pred]
    )[0]

    return queue


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Ticket Classifier",
        "Model Analysis"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title(
        "🤖 Customer Support Ticket Automation"
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Size",
        "28,260"
    )

    c2.metric(
        "Queues",
        "10"
    )

    c3.metric(
        "Best Accuracy",
        "75.9%"
    )

    c4.metric(
        "Best Macro F1",
        "78.8%"
    )

    st.markdown("---")

    st.subheader("Project Workflow")

    st.markdown(
        """
        Customer Ticket
            ↓
        DistilBERT Classification
            ↓
        Queue Prediction
            ↓
        Gemini 2.5 Flash
            ↓
        Customer Acknowledgement
        """
    )

    st.markdown("---")

    st.subheader("Business Benefits")

    st.markdown("""
    - Automated Ticket Routing
    - Faster Resolution
    - Reduced Manual Effort
    - Instant Customer Acknowledgement
    - Improved Customer Experience
    - Scalable Support Operations
    """)

# =====================================================
# CLASSIFIER
# =====================================================

elif page == "Ticket Classifier":

    st.title("🎫 Ticket Classification")

    ticket_text = st.text_area(
        "Enter Customer Ticket",
        height=250
    )

    if st.button("Classify Ticket"):

        if len(ticket_text.strip()) == 0:

            st.warning(
                "Please enter ticket text."
            )

        else:

            with st.spinner(
                "Predicting Queue..."
            ):

                queue = predict_queue(
                    ticket_text
                )

            st.success(
                f"Predicted Queue: {queue}"
            )

            with st.spinner(
                "Generating Response..."
            ):

                reply = gemini.generate_response(
                    ticket_text,
                    queue
                )

            st.subheader(
                "Generated Customer Reply"
            )

            st.info(reply)

# =====================================================
# MODEL ANALYSIS
# =====================================================

elif page == "Model Analysis":

    st.title("📊 Model Comparison")

    results = pd.DataFrame({

        "Model": [
            "Naive Bayes",
            "Logistic Regression",
            "Linear SVM",
            "BiLSTM",
            "DistilBERT"
        ],

        "Accuracy": [
            42.99,
            58.51,
            70.84,
            74.27,
            75.90
        ],

        "Macro F1": [
            23.47,
            49.13,
            71.96,
            75.15,
            78.80
        ]
    })

    st.dataframe(
        results,
        use_container_width=True
    )

    st.bar_chart(
        results.set_index("Model")
    )

    st.markdown("---")

    st.subheader(
        "Why Accuracy is not Higher?"
    )

    st.markdown("""
    ### 1. Overlapping Classes

    Technical Support and IT Support often contain very similar ticket content.

    ### 2. Class Imbalance

    Some queues contain far fewer examples than others.

    ### 3. Ambiguous Tickets

    Many tickets could belong to multiple departments.

    ### 4. Label Noise

    Human-assigned queue labels are not always consistent.

    ### 5. Long Ticket Truncation

    DistilBERT only processes the first 256 tokens.

    ### 6. DistilBERT Tradeoff

    DistilBERT sacrifices a small amount of accuracy
    for faster inference and lower memory usage.
    """)

    st.markdown("---")

    st.subheader(
        "Most Common Confusions"
    )

    st.markdown("""
    - Technical Support ↔ IT Support
    - Customer Service ↔ Product Support
    - Billing ↔ Returns & Exchanges
    """)

    st.markdown("---")

    st.subheader(
        "Future Improvements"
    )

    st.markdown("""
    - RoBERTa Fine Tuning
    - DeBERTa V3
    - Ensemble Models
    - Sentiment Analysis
    - Priority Prediction
    - Active Learning
    - Multi-label Classification
    """)

