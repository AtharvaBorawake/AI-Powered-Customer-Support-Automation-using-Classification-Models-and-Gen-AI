"""
explore_data.py

Performs exploratory data analysis (EDA) on the
Customer Support Tickets dataset.
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data
from preprocessing import clean_text


def main():

    df = load_data()

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Total Samples : {len(df)}")
    print(f"Number of Queues : {df['queue'].nunique()}")
    print(f"Languages : {df['language'].unique()}")
    print()

    # ----------------------------
    # Queue Distribution
    # ----------------------------

    print("=" * 60)
    print("QUEUE DISTRIBUTION")
    print("=" * 60)

    queue_counts = df["queue"].value_counts()

    print(queue_counts)
    print()

    # ----------------------------
    # Priority Distribution
    # ----------------------------

    print("=" * 60)
    print("PRIORITY DISTRIBUTION")
    print("=" * 60)

    priority_counts = df["priority"].value_counts()

    print(priority_counts)
    print()

    # ----------------------------
    # Ticket Length
    # ----------------------------

    lengths = df["text"].apply(lambda x: len(str(x).split()))

    print("=" * 60)
    print("TICKET LENGTH STATISTICS")
    print("=" * 60)

    print(f"Mean   : {lengths.mean():.2f}")
    print(f"Median : {lengths.median():.2f}")
    print(f"Min    : {lengths.min()}")
    print(f"Max    : {lengths.max()}")
    print(f"95%    : {np.percentile(lengths,95):.2f}")

    print()

    # ----------------------------
    # Most Common Words
    # ----------------------------

    print("=" * 60)
    print("TOP 20 WORDS")
    print("=" * 60)

    all_words = []

    for text in df["text"]:

        cleaned = clean_text(text)

        all_words.extend(cleaned.split())

    counter = Counter(all_words)

    for word, count in counter.most_common(20):
        print(f"{word:20} {count}")

    # ----------------------------
    # Plots
    # ----------------------------

    plt.figure(figsize=(10, 5))
    queue_counts.plot(kind="bar")
    plt.title("Queue Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs_queue_distribution.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    priority_counts.plot(kind="bar")
    plt.title("Priority Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs_priority_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=40)
    plt.title("Ticket Length Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs_ticket_length_histogram.png")
    plt.close()

    print()
    print("=" * 60)
    print("Plots Saved Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()