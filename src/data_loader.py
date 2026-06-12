"""
data_loader.py

Loads the customer support ticket dataset and prepares
a pandas DataFrame for further processing.
"""

import pandas as pd


DATASET_NAME = "Tobi-Bueck/customer-support-tickets"


def load_data(language="en"):
    """
    Load dataset from Hugging Face and return a cleaned DataFrame.

    Parameters
    ----------
    language : str
        Language filter (default = "en")

    Returns
    -------
    pandas.DataFrame
    """
    print("1. Importing datasets...")
    from datasets import load_dataset

    print("2. Calling load_dataset...")
    dataset = load_dataset(DATASET_NAME)
    

    print("3. Converting to pandas...")
    df = dataset["train"].to_pandas()

    print("4. Filtering...")
    df = df[df["language"] == language].copy()

    print("5. Dropping NA...")
    df = df.dropna(subset=["body", "queue"])

    print("6. Filling subject...")
    df["subject"] = df["subject"].fillna("")

    print("7. Combining text...")
    df["text"] = df["subject"].str.strip() + " " + df["body"].str.strip()

    print("8. Cleaning spaces...")
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

    df.reset_index(drop=True, inplace=True)

    print("9. Returning dataframe")

    return df


if __name__ == "__main__":
    df = load_data()

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Total Samples : {len(df)}")
    print(f"Languages     : {df['language'].unique()}")
    print(f"Queues        : {df['queue'].nunique()}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample Text:\n")
    print(df.loc[0, "text"][:500])