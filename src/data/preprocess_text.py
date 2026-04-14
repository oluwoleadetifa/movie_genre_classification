# src/data/preprocess_text.py

import re
import pandas as pd


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)      
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text_dataframe(
    df: pd.DataFrame,
    text_column: str = "overview",
    output_column: str = "clean_text"
) -> pd.DataFrame:
    df = df.copy()
    df[text_column] = df[text_column].fillna("")
    df[output_column] = df[text_column].apply(clean_text)
    return df
