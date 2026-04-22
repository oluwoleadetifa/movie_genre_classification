import re
import pandas as pd


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text_dataframe(
    df,
    text_column="description",
    output_column="clean_text"
):
    df = df.copy()
    df[text_column] = df[text_column].fillna("")
    df[output_column] = df[text_column].apply(clean_text)

    return df
