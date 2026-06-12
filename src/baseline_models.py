"""
baseline_models.py

Train and compare baseline ML models:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVM
"""

import os

import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from data_loader import load_data
from preprocessing import clean_text


# --------------------------
# Load dataset
# --------------------------

df = load_data()

print("Cleaning text...")

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["queue"]


# --------------------------
# Train Test Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# --------------------------
# TF-IDF
# --------------------------

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# --------------------------
# Models
# --------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC(class_weight="balanced")

}


best_model = None
best_score = 0


os.makedirs("models", exist_ok=True)


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)

    macro_f1 = f1_score(
        y_test,
        preds,
        average="macro"
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print()

    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    )

    plt.figure(figsize=(10, 10))

    disp.plot(
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False,
    )

    plt.title(name)

    plt.tight_layout()

    plt.savefig(
        f"{name.replace(' ','_')}_confusion_matrix.png"
    )

    plt.close()

    if macro_f1 > best_score:

        best_score = macro_f1

        best_model = model


joblib.dump(best_model, "models/best_baseline_model.pkl")

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")


print("\n")
print("=" * 60)
print("Best baseline model saved successfully.")
print("=" * 60)