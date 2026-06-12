"""
feature_engineering.py

Creates additional numerical features for customer support tickets.

These features can be concatenated with TF-IDF vectors for
classical machine learning models.
"""

import re

from textblob import TextBlob


# ----------------------------
# Urgency keywords
# ----------------------------

URGENCY_WORDS = {
    "urgent",
    "asap",
    "immediately",
    "critical",
    "emergency",
    "important",
    "priority",
    "help",
    "quickly"
}


# ----------------------------
# Business keywords
# ----------------------------

KEYWORDS = {
    "refund",
    "payment",
    "invoice",
    "billing",
    "login",
    "password",
    "account",
    "error",
    "bug",
    "crash",
    "exchange",
    "return"
}


def extract_features(text: str) -> dict:
    """
    Extract handcrafted numerical features from text.
    """

    if not isinstance(text, str):
        text = ""

    lower = text.lower()

    words = re.findall(r"\b[a-z]+\b", lower)

    # -----------------------
    # Basic statistics
    # -----------------------

    word_count = len(words)

    char_count = len(text)

    avg_word_length = (
        sum(len(w) for w in words) / word_count
        if word_count > 0 else 0
    )

    # -----------------------
    # Urgency
    # -----------------------

    urgency_score = sum(
        1 for w in words
        if w in URGENCY_WORDS
    )

    # -----------------------
    # Keyword count
    # -----------------------

    keyword_score = sum(
        1 for w in words
        if w in KEYWORDS
    )

    # -----------------------
    # Exclamation marks
    # -----------------------

    exclamation_count = text.count("!")

    # -----------------------
    # Question marks
    # -----------------------

    question_count = text.count("?")

    # -----------------------
    # Uppercase ratio
    # -----------------------

    uppercase_chars = sum(c.isupper() for c in text)

    uppercase_ratio = (
        uppercase_chars / len(text)
        if len(text) > 0 else 0
    )

    # -----------------------
    # Sentiment
    # -----------------------

    sentiment = TextBlob(text).sentiment.polarity

    return {

        "word_count": word_count,

        "char_count": char_count,

        "avg_word_length": avg_word_length,

        "urgency_score": urgency_score,

        "keyword_score": keyword_score,

        "exclamation_count": exclamation_count,

        "question_count": question_count,

        "uppercase_ratio": uppercase_ratio,

        "sentiment": sentiment,
    }


if __name__ == "__main__":

    sample = """
    URGENT!!

    I cannot login to my account.

    My payment failed and I need help ASAP!!

    """

    features = extract_features(sample)

    print("=" * 50)

    for k, v in features.items():
        print(f"{k:20} : {v}")