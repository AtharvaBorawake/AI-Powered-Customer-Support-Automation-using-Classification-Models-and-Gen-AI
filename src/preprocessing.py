"""
preprocessing.py

Contains text preprocessing functions for:
1. Classical Machine Learning models (TF-IDF, Logistic Regression, SVM, NB)
2. Transformer models (DistilBERT/BERT)
"""

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources only once
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOPWORDS = set(stopwords.words("english"))

# Keep negation words
STOPWORDS -= {
    "no",
    "not",
    "nor"
}
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Heavy preprocessing for classical ML models.
    """

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # tokenize
    words = text.split()

    words = [
    LEMMATIZER.lemmatize(word)
    for word in words
    if word not in STOPWORDS
    ]

    return " ".join(words)


def minimal_clean(text: str) -> str:
    """
    Minimal preprocessing for transformer models.
    DistilBERT should receive text close to its original form.
    """

    if not isinstance(text, str):
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


if __name__ == "__main__":

    sample = """
 Dear Customer Support Team,\n\nI am reaching out to seek guidance on implementing vital upgrades to our infrastructure, which are crucial for advancing the company's data processing capabilities. The objectives include improving support for sophisticated data lakes, incorporating Python-based machine learning models, and deploying automated balancing engines across various divisions. These improvements aim to enhance real-time analytical functions, ESG (Environmental, Social, Governance) reporting, and automation of investment strategies.\n\nAt present, our infrastructure does not support scalability.
    """

    print("=" * 60)
    print("Original:\n")
    print(sample)

    print("\n" + "=" * 60)
    print("Classical ML Cleaning:\n")
    print(clean_text(sample))

    print("\n" + "=" * 60)
    print("Transformer Cleaning:\n")
    print(minimal_clean(sample))